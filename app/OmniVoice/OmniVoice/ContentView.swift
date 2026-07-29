//
//  ContentView.swift
//  OmniVoice
//
//  Created by Rory Clear on 22/07/2026.
//

import SwiftUI
import Foundation

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
var buffers: [Int: MTLBuffer] = [:]
var buffer_sz: [Int: Int] = [:] // todo
var programs: [String: MTLComputePipelineState] = [:]
var encode_graph: GraphRunner!
var model_graph: GraphRunner!
let CHAR_WEIGHTS = try! JSONDecoder().decode([Float].self, from: Data(contentsOf: Bundle.main.url(forResource: "char_weights", withExtension: "json")!))
let AUDIO_CHUNK_DURATION = 15.0
let FRAME_RATE = 25
let AUDIO_MASK_ID = 1024
let MAX_LEN = 750
let NUM_AUDIO_CODEBOOK = 8
let T_SHIFT = 0.1
let SAMPLING_RATE = 24_000
let CHUNK_SIZE = 960
let tokenizer = Tokenizer()

class Tokenizer {
    let specialTokens: [String: Int]
    let normalTokensBytes: [[UInt8]: Int]

    private let byteDecoder: [Character: UInt8]

    init() {
        let url = Bundle.main.url(forResource: "tokenizer", withExtension: "json")!
        let data = try! Data(contentsOf: url)
        let json = try! JSONSerialization.jsonObject(with: data) as! [String: Any]

        // Build locally first
        var decoder: [Character: UInt8] = [:]

        var bs = Array(33...126)
        bs += Array(161...172)
        bs += Array(174...255)

        var cs = bs
        var n = 0

        for b in 0..<256 {
            if !bs.contains(b) {
                bs.append(b)
                cs.append(256 + n)
                n += 1
            }
        }

        for (b, c) in zip(bs, cs) {
            decoder[Character(UnicodeScalar(c)!)] = UInt8(b)
        }

        // Special tokens
        let addedTokens = json["added_tokens"] as! [[String: Any]]

        let specials = Dictionary(
            uniqueKeysWithValues: addedTokens.map {
                (
                    $0["content"] as! String,
                    $0["id"] as! Int
                )
            }
        )

        // Normal vocab
        let model = json["model"] as! [String: Any]
        let vocab = model["vocab"] as! [String: Int]

        let normal = Dictionary(
            uniqueKeysWithValues: vocab.map { token, id in
                (
                    token.map { decoder[$0]! },
                    id
                )
            }
        )

        // Assign all properties last
        self.byteDecoder = decoder
        self.specialTokens = specials
        self.normalTokensBytes = normal
    }


    func encode(_ text: String) -> [Int] {
        var result: [Int] = []

        var remaining = text

        let specials = specialTokens.keys.sorted {
            $0.count > $1.count
        }

        while !remaining.isEmpty {

            if let special = specials.first(where: {
                remaining.hasPrefix($0)
            }) {
                result.append(specialTokens[special]!)
                remaining.removeFirst(special.count)
                continue
            }

            // encode until next possible special token
            var chunk = remaining

            if let next = specials.compactMap({
                remaining.range(of: $0)?.lowerBound
            }).min() {
                chunk = String(remaining[..<next])
                remaining = String(remaining[next...])
            } else {
                remaining = ""
            }

            result += encodeWord(chunk)
        }

        return result
    }


    private func encodeWord(_ word: String) -> [Int] {
        let bytes = Array(word.utf8)

        if let id = normalTokensBytes[bytes] {
            return [id]
        }

        var parts = bytes.map {
            [$0]
        }

        while true {
            var bestID = Int.max
            var bestIndex = -1

            for i in 0..<(parts.count - 1) {
                let merged = parts[i] + parts[i + 1]

                if let id = normalTokensBytes[merged], id < bestID {
                    bestID = id
                    bestIndex = i
                }
            }

            if bestIndex == -1 {
                break
            }

            parts[bestIndex] += parts[bestIndex + 1]
            parts.remove(at: bestIndex + 1)
        }

        return parts.map {
            normalTokensBytes[$0]!
        }
    }
}

class GraphRunner {
    let filename: String
    var calls: [[String: Any]] = []
    var copyouts: [Int] = []
    var copyins: [Int] = []
    var buffs: Set<Int> = []

    init(filename: String) {
        self.filename = filename
        print("GraphRunner initialized with:", filename)
        loadFile()
    }

    private func loadFile() {
        guard let url = Bundle.main.url(forResource: filename, withExtension: nil) else {
            print("File not found:", filename)
            return
        }

        autoreleasepool {
            do {
                var fileData: Data? = try Data(contentsOf: url)
                guard let data = fileData else { return }
                let json = try JSONSerialization.jsonObject(with: data, options: [])
                fileData = nil

                guard let items = json as? [Any] else {
                    print("Invalid JSON format")
                    return
                }

                for item in items {
                    autoreleasepool {
                        guard let dict = item as? [String: Any],
                              let key = dict.keys.first else {
                            return
                        }

                        if key == "buff_alloc" {
                            if let info = dict["buff_alloc"] as? [String: Any],
                               let num = info["num"] as? Int,
                               let size = info["size"] as? Int {

                                buffers[num] = device.makeBuffer(
                                    length: size,
                                    options: .storageModeShared
                                )

                                buffer_sz[num] = size
                            }

                        } else if key == "copyin" {
                            if let info = dict["copyin"] as? [String: Any],
                               let dest = info["dest"] as? Int,
                               let dataString = info["data"] as? String,
                               let decodedData = Data(base64Encoded: dataString),
                               let buffer = buffers[dest] {

                                copyins.append(dest)

                                let ptr = buffer.contents()
                                decodedData.copyBytes(
                                    to: ptr.assumingMemoryBound(to: UInt8.self),
                                    count: decodedData.count
                                )
                            }

                        } else if key == "program" {
                            if let info = dict["program"] as? [String: Any],
                               let name = info["name"] as? String,
                               let libString = info["lib"] as? String,
                               let libData = Data(base64Encoded: libString) {

                                let dispatchData = libData.withUnsafeBytes { ptr in
                                    DispatchData(bytes: ptr)
                                }

                                if let library = try? device.makeLibrary(
                                    data: dispatchData as! dispatch_data_t
                                ),
                                let function = library.makeFunction(name: name),
                                let pipeline = try? device.makeComputePipelineState(
                                    function: function
                                ) {
                                    programs[name] = pipeline
                                }
                            }

                        } else if key == "call" {
                            if let call = dict["call"] as? [String: Any] {
                                calls.append(call)
                                for buff in call["buffers"] as! [Int] {
                                    buffs.insert(buff)
                                }
                            }
                        } else if key == "copyout" {
                            if let copyout = dict["copyout"] as? Int {
                                copyouts.append(copyout)
                            }
                        }
                    }
                }
            } catch {
                print("Failed reading JSON:", error)
            }
        }
    }
    
    func run(vals_dict: [Int: Int]? = nil) {
        for (index, item) in self.calls.enumerated() {
            let commandBuffer = queue.makeCommandBuffer()!
            let encoder = commandBuffer.makeComputeCommandEncoder()!
            print(index, "of", self.calls.count)
            let name = item["name"] as! String
            print(name)
            let pipeline = programs[name]!

            encoder.setComputePipelineState(pipeline)

            let bufferIDs = item["buffers"] as! [Int]
            let offsets = item["buffer_offsets"] as! [Int]
            let vals = item["vals"] as! [Int]
            print("vals =", vals)

            for i in 0..<bufferIDs.count {
                let buffer = buffers[bufferIDs[i]]!
                encoder.setBuffer(buffer, offset: offsets[i], index: i)
            }
            
            for i in 0..<vals.count{
                var value = Int32(vals_dict![vals[i]]!)
                encoder.setBytes(&value, length: 4, index: i+bufferIDs.count)
            }
            
            let global = item["global_size"] as! [Int]
            let local = item["local_size"] as! [Int]

            let threadsPerGrid = MTLSize(
                width: global[0],
                height: global[1],
                depth: global[2]
            )

            let threadsPerThreadgroup = MTLSize(
                width: local[0],
                height: local[1],
                depth: local[2]
            )

            encoder.dispatchThreadgroups(
                threadsPerGrid,
                threadsPerThreadgroup: threadsPerThreadgroup
            )
            encoder.endEncoding()
            // todo, all at once is better probably, with just one command buffer init, this is nice to watch though
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

        }
    }
    
}

struct ContentView: View {
    var body: some View {
        VStack {
            Image(systemName: "globe")
                .imageScale(.large)
                .foregroundStyle(.tint)
            Text("Hello, world!")
        }
        .padding()
        .onAppear {
            //run_tests()
            encode_graph = GraphRunner(filename: "0.rc")
            model_graph = GraphRunner(filename: "1.rc")
            generate(
                text: "Testing testing one two three, this is made with Omni-Voice. Can you hear me? or not? thank you for listening to this",
                refText: "This is a wav file for my voice, so that omni voice can capture my voice. I need to talk for about 15 seconds",
                file: "voice4_short",
                num_steps: 16,
                language: "None"
            )
        }
    }
}

func generate(text: String, refText: String, file: String, num_steps: Int, language: String) {
    var ref_wav = load_audio(file: file, samplingRate: 24000)
    let wav_len = ref_wav.count
    ref_wav = expandWav(ref_wav)
    memcpy(buffers[encode_graph.copyins.last!]!.contents(), ref_wav, ref_wav.count * MemoryLayout<Float>.stride)
    encode_graph.run()
    let ref_audio_tokens = get_ref_tokens()
    print(ref_audio_tokens)
}


func load_audio(file: String, samplingRate: Int = SAMPLING_RATE) -> [Float] {
    let url = Bundle.main.url(forResource: file, withExtension: "wav")!
    let audio = try! Data(contentsOf: url)
    let (data, sr) = load_waveform(audio)

    // Convert multi-channel to mono
    let sampleCount = data[0].count
    var mono = [Float]()
    mono.reserveCapacity(sampleCount)

    for i in 0..<sampleCount {
        var sum: Float = 0
        for channel in data {
            sum += channel[i]
        }
        mono.append(sum / Float(data.count))
    }

    // Resample
    let resampled = resample([mono], origSR: sr, targetSR: samplingRate)[0]

    // RMS
    let rms = sqrt(
        resampled.reduce(0.0) { $0 + Double($1 * $1) } /
        Double(resampled.count)
    )

    print("rms =", rms)

    var output = resampled

    if rms > 0 && rms < 0.1 {
        let scale = Float(0.1 / rms)
        output = output.map { $0 * scale }
    }

    return output
}

func expandWav(_ refWav: [Float]) -> [Float] {
    var refWav = refWav
    let clipSize = refWav.count % CHUNK_SIZE

    if clipSize > 0 {
        refWav = Array(refWav.dropLast(clipSize))
    }

    let wavLen = refWav.count
    let targetLen = SAMPLING_RATE * 20

    if wavLen <= targetLen {
        refWav += Array(repeating: 0.0, count: targetLen - wavLen)
    } else {
        refWav = Array(refWav.prefix(targetLen))
    }

    return refWav
}

func resample(
    _ data: [[Float]],
    origSR: Int,
    targetSR: Int
) -> [[Float]] {
    let sampleCount = data[0].count

    let duration = Double(sampleCount) / Double(origSR)

    let origTimes = (0..<sampleCount).map {
        Double($0) * duration / Double(sampleCount)
    }

    let newLength = Int(duration * Double(targetSR))

    let newTimes = (0..<newLength).map {
        Double($0) * duration / Double(newLength)
    }

    return data.map { channel in
        interp1D(
            newTimes: newTimes,
            origTimes: origTimes,
            values: channel
        )
    }
}

func interp1D(
    newTimes: [Double],
    origTimes: [Double],
    values: [Float]
) -> [Float] {
    var result = [Float]()
    result.reserveCapacity(newTimes.count)

    var index = 0

    for t in newTimes {
        while index < origTimes.count - 2 &&
              origTimes[index + 1] < t {
            index += 1
        }

        let t0 = origTimes[index]
        let t1 = origTimes[index + 1]

        let y0 = Double(values[index])
        let y1 = Double(values[index + 1])

        let ratio = (t - t0) / (t1 - t0)

        result.append(
            Float(y0 + ratio * (y1 - y0))
        )
    }

    return result
}

func load_waveform(_ data: Data) -> ([[Float]], Int) {
    let bytes = [UInt8](data)
    let sampleRate = Int(readUInt32LE(bytes, offset: 24))
    let channels = Int(readUInt16LE(bytes, offset: 22))
    let dataMarker = [UInt8]([100, 97, 116, 97]) // "data"
    var dataOffset = 0

    for i in 0..<(bytes.count - 4) {
        if Array(bytes[i..<i+4]) == dataMarker {
            dataOffset = i + 8
            break
        }
    }

    let rawSamples = Array(bytes[dataOffset..<bytes.count])

    // int16 = 2 bytes
    let nSamples = rawSamples.count / 2

    var samples = [Int16]()
    samples.reserveCapacity(nSamples)

    for i in stride(from: 0, to: rawSamples.count, by: 2) {
        let sample = Int16(bitPattern:
            UInt16(rawSamples[i]) |
            (UInt16(rawSamples[i + 1]) << 8)
        )
        samples.append(sample)
    }

    // Create channel arrays
    var audio = Array(
        repeating: [Float](),
        count: channels
    )

    for i in stride(from: 0, to: samples.count, by: channels) {
        for ch in 0..<channels {
            if i + ch < samples.count {
                audio[ch].append(
                    Float(samples[i + ch]) / 32768.0
                )
            }
        }
    }

    return (audio, sampleRate)
}

func readUInt16LE(_ bytes: [UInt8], offset: Int) -> UInt16 {
    return UInt16(bytes[offset]) |
           (UInt16(bytes[offset + 1]) << 8)
}

func readUInt32LE(_ bytes: [UInt8], offset: Int) -> UInt32 {
    return UInt32(bytes[offset]) |
           (UInt32(bytes[offset + 1]) << 8) |
           (UInt32(bytes[offset + 2]) << 16) |
           (UInt32(bytes[offset + 3]) << 24)
}

#Preview {
    ContentView()
}

func get_ref_tokens() -> [[Int32]] {
    let count = buffer_sz[encode_graph.copyouts[0]]!
    let flat = Array(UnsafeBufferPointer(start: buffers[encode_graph.copyouts[0]]!.contents().assumingMemoryBound(to: Int32.self), count: count))
    let prefixCount = Int((20 * SAMPLING_RATE * NUM_AUDIO_CODEBOOK) / CHUNK_SIZE)
    let trimmed = Array(flat.prefix(prefixCount))
    let n = trimmed.count / NUM_AUDIO_CODEBOOK
    return stride(from: 0, to: trimmed.count, by: n).map { Array(trimmed[$0..<($0 + n)]) }
}

//todo......
func run_tests() {
    //audio load
    var value = load_audio(file: "voice3")
    var expected = (try! JSONDecoder().decode([Float].self, from: Data(contentsOf: Bundle.main.url(forResource: "voice3_ref_wav", withExtension: "json")!)))
    assert(value.count == expected.count && zip(value, expected).allSatisfy { abs($0 - $1) < 1e-5 })
    value = load_audio(file: "voice4")
    expected = (try! JSONDecoder().decode([Float].self, from: Data(contentsOf: Bundle.main.url(forResource: "voice4_ref_wav", withExtension: "json")!)))
    assert(value.count == expected.count && zip(value, expected).allSatisfy { abs($0 - $1) < 1e-5 })
    
    // expand to 20s
    value = try! JSONDecoder().decode([Float32].self, from: Data(contentsOf: Bundle.main.url(forResource: "voice3_ref_wav", withExtension: "json")!))
    value = expandWav(value)
    expected = (try! JSONDecoder().decode([Float].self, from: Data(contentsOf: Bundle.main.url(forResource: "voice3_ref_wav_exp", withExtension: "json")!)))
    assert(value.count == expected.count && zip(value, expected).allSatisfy { abs($0 - $1) < 1e-5 })
    value = try! JSONDecoder().decode([Float32].self, from: Data(contentsOf: Bundle.main.url(forResource: "voice4_ref_wav", withExtension: "json")!))
    value = expandWav(value)
    expected = (try! JSONDecoder().decode([Float].self, from: Data(contentsOf: Bundle.main.url(forResource: "voice4_ref_wav_exp", withExtension: "json")!)))
    assert(value.count == expected.count && zip(value, expected).allSatisfy { abs($0 - $1) < 1e-5 })
    
    // encode
    encode_graph = GraphRunner(filename: "0.rc")
    value = (try! JSONDecoder().decode([Float].self, from: Data(contentsOf: Bundle.main.url(forResource: "voice4_ref_wav_exp", withExtension: "json")!)))
    memcpy(buffers[encode_graph.copyins.last!]!.contents(), value, value.count * MemoryLayout<Float>.stride)
    encode_graph.run()
    var out = get_ref_tokens()
    var expected_tokens = try! JSONDecoder().decode([[[Int32]]].self, from: Data(contentsOf: Bundle.main.url(forResource: "voice4_ref_audio_tokens", withExtension: "json")!))[0]
    assert(out == expected_tokens, "Token mismatch: got \(out), expected \(expected_tokens)")
    
    value = (try! JSONDecoder().decode([Float].self, from: Data(contentsOf: Bundle.main.url(forResource: "voice4_ref_wav_exp", withExtension: "json")!)))
    memcpy(buffers[encode_graph.copyins.last!]!.contents(), value, value.count * MemoryLayout<Float>.stride)
    encode_graph.run()
    out = get_ref_tokens()
    expected_tokens = try! JSONDecoder().decode([[[Int32]]].self, from: Data(contentsOf: Bundle.main.url(forResource: "voice4_ref_audio_tokens", withExtension: "json")!))[0]
    assert(out == expected_tokens, "Token mismatch: got \(out), expected \(expected_tokens)")
    

    print("DONE")
}


