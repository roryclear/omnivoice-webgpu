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
let commandBuffer = queue.makeCommandBuffer()!
var encode_graph: GraphRunner!
let CHAR_WEIGHTS = try! JSONDecoder().decode([Float].self, from: Data(contentsOf: Bundle.main.url(forResource: "char_weights", withExtension: "json")!))
let AUDIO_CHUNK_DURATION = 15.0
let FRAME_RATE = 25
let AUDIO_MASK_ID = 1024
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
        do {
            let data = try Data(contentsOf: url)
            let json = try JSONSerialization.jsonObject(with: data, options: [])

            let items = json as! [Any]

            //print("Items:", items.count)

            for (_, item) in items.enumerated() {
                let dict = item as! [String: Any]
                let key = dict.keys.first!

                if key == "buff_alloc" {
                    if let info = dict["buff_alloc"] as? [String: Any],
                       let num = info["num"] as? Int,
                       let size = info["size"] as? Int {
                        buffers[num] = MTLCreateSystemDefaultDevice()?.makeBuffer(length: size, options: .storageModeShared)
                        buffer_sz[num] = size
                    }
                } else if key == "copyin" {
                    if let info = dict["copyin"] as? [String: Any],
                       let dest = info["dest"] as? Int,
                       let dataString = info["data"] as? String,
                       let data = Data(base64Encoded: dataString),
                       let buffer = buffers[dest] {
                        self.copyins.append(dest)
                        let ptr = buffer.contents()
                        data.copyBytes(to: ptr.assumingMemoryBound(to: UInt8.self), count: data.count)
                    }
                } else if key == "program" {
                    if let info = dict["program"] as? [String: Any],
                       let name = info["name"] as? String,
                       let libString = info["lib"] as? String,
                       let libData = Data(base64Encoded: libString),
                       let device = MTLCreateSystemDefaultDevice() {
                        let dispatchData = libData.withUnsafeBytes { ptr in
                            DispatchData(bytes: ptr)
                        }
                        if let library = try? device.makeLibrary(data: dispatchData as! dispatch_data_t) {
                            if let function = library.makeFunction(name: name) {
                                if let pipeline = try? device.makeComputePipelineState(function: function) {
                                    programs[name] = pipeline
                                }
                            }
                        }
                    }
                } else if key == "call" {
                    calls.append(dict["call"] as! [String: Any])
                } else if key == "copyout" {
                    copyouts.append(dict["copyout"] as! Int)
                }
            }
        } catch {
            print("Failed reading JSON:", error)
        }
    }
    
    func run() {
        for item in self.calls {
            let name = item["name"] as! String
            let pipeline = programs[name]!

            let encoder = commandBuffer.makeComputeCommandEncoder()!

            encoder.setComputePipelineState(pipeline)

            let bufferIDs = item["buffers"] as! [Int]
            let offsets = item["buffer_offsets"] as! [Int]


            for i in 0..<bufferIDs.count {
                let buffer = buffers[bufferIDs[i]]!
                encoder.setBuffer(buffer, offset: offsets[i], index: i)
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
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
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
            encode_graph = GraphRunner(filename: "0.rc")
            runGenerate()
        }
    }

    func runGenerate() {
        let url = Bundle.main.url(forResource: "voice4", withExtension: "wav")!
        let audioData = try! Data(contentsOf: url)
        generate(
            text: "Testing testing one two three, this is made with Omni-Voice. Can you hear me? or not? thank you for listening to this",
            refText: "This is a wav file for my voice, so that omni voice can capture my voice. I need to talk for about 15 seconds emm we're on about eleven right now, so I just need to say a few more words, thank you",
            refAudio: audioData,
            numSteps: 16,
            language: "en"
        )
    }
}

func estimateTargetTokens(_ text: String, _ refText: String, _ numRefAudioTokens: Int) -> Int {
    let refWeight = refText.unicodeScalars.reduce(Float(0)) { $0 + CHAR_WEIGHTS[Int($1.value)] }
    let speedFactor = refWeight / Float(numRefAudioTokens)
    let targetWeight = text.unicodeScalars.reduce(Float(0)) { $0 + CHAR_WEIGHTS[Int($1.value)] }
    let estimatedDuration = targetWeight / speedFactor
    return Int(estimatedDuration)
}

func generate(
    text: String,
    refText: String,
    refAudio: Data,
    numSteps: Int = 16,
    language: String = "None"
) {
    let sampling_rate = 24000
    let chunk_size = 960
    var ref_wav = load_audio(refAudio, samplingRate: sampling_rate)
    
    
    let clipSize = ref_wav.count % chunk_size
    if clipSize > 0 { ref_wav.removeLast(clipSize) }
    let wavLen = ref_wav.count
    if wavLen < (sampling_rate * 20) { ref_wav.append(contentsOf: Array(repeating: 0.0, count: (sampling_rate * 20) - wavLen)) }
    
    let numRefAudioTokens = wavLen / chunk_size

    var targetLength = estimateTargetTokens(text, refText, numRefAudioTokens)

    print("target_length =", targetLength)
    
    ref_wav.withUnsafeBytes { memcpy(buffers[encode_graph.copyins[encode_graph.copyins.count - 1]]!.contents(), $0.baseAddress!, ref_wav.count * MemoryLayout<Float>.size) }
    
    let avgTokensPerChar = Float(targetLength) / Float(text.count)
    let textChunkLen = Int(Float(AUDIO_CHUNK_DURATION) * Float(FRAME_RATE) / avgTokensPerChar)

    let pattern = #"[^。，！？；：、.,?]+[。，！？；：、.,?]?"#
    let regex = try! NSRegularExpression(pattern: pattern)
    let range = NSRange(text.startIndex..., in: text)

    let chunksSmall = regex.matches(in: text, range: range).compactMap {
        Range($0.range, in: text).map { String(text[$0]) }
    }

    var chunks = [""]
    var j = 0

    for var chunk in chunksSmall {
        if chunk.first == " " {
            chunk.removeFirst()
        }

        if chunks[j].count < textChunkLen + chunk.count {
            chunks[j] += chunk
        } else {
            chunks.append(chunk)
            j += 1
        }
    }
    
    print("CHUNKS", chunks.count)
    print(chunks)
    
    let start = ContinuousClock.now
    encode_graph.run()
    let elapsed = start.duration(to: .now)
    print("Execution time: \(elapsed)")
    
    let data = Data(bytes: buffers[encode_graph.copyouts[0]]!.contents(), count: buffer_sz[encode_graph.copyouts[0]]!)
    let first1000Bytes = data.prefix(1000)

    let intCount = first1000Bytes.count / MemoryLayout<Int32>.size

    let intArray: [Int32] = first1000Bytes.withUnsafeBytes { ptr in
        let buffer = ptr.bindMemory(to: Int32.self)
        return Array(buffer.prefix(intCount))
    }
    //todo, if this doesn't change fix
    print("size =",buffer_sz[encode_graph.copyouts[0]]!)
    
    let ints = data.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
    let cols = ints.count / 8
    let refAudioTokens = stride(from: 0, to: numRefAudioTokens * 8, by: 8).map { Array(ints[$0..<$0 + 8]) }

    print("refAudioTokens:",refAudioTokens)
    for chunk in chunks {
        targetLength = estimateTargetTokens(chunk, refText, numRefAudioTokens)
        generateIterative(chunk, targetLength: targetLength, refText: refText, refAudioTokens: refAudioTokens)
    }
    
}

func generateIterative(_ text: String, targetLength: Int, refText: String, refAudioTokens: [[Int32]], numSteps: Int = 16, language: String = "None") {
    let style_tokens = tokenizer.encode("<|denoise|><|lang_start|>\(language)<|lang_end|><|instruct_start|>None<|instruct_end|>")
    let text_tokens = tokenizer.encode("<|text_start|>\(( [refText, text].map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }.joined(separator: " ")))<|text_end|>")
    let target_audio_tokens = Array(repeating: AUDIO_MASK_ID, count: targetLength)
    let c_len = style_tokens.count + text_tokens.count + refAudioTokens.count + target_audio_tokens.count
    let cond_audio_start_idx = c_len - targetLength - refAudioTokens.count
    print("c_len =",c_len, "cond_audio_start_idx", cond_audio_start_idx)
}

func load_audio(_ audio: Data, samplingRate: Int) -> [Float] {
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


