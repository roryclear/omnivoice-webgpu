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
let MAX_LEN = 500
let NUM_AUDIO_CODEBOOK = 8
let T_SHIFT = 0.1
let SAMPLING_RATE = 24_000
let CHUNK_SIZE = 960
let REF_AUDIO_LEN = 10
let tokenizer = Tokenizer()

class Tokenizer {
    let specialTokens: [String: Int32]
    let normalTokensBytes: [[UInt8]: Int32]

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
                    $0["id"] as! Int32
                )
            }
        )

        // Normal vocab
        let model = json["model"] as! [String: Any]
        let vocab = model["vocab"] as! [String: Int32]

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


    func encode(_ text: String) -> [Int32] {
        var result: [Int32] = []

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


    private func encodeWord(_ word: String) -> [Int32] {
        let bytes = Array(word.utf8)

        if let id = normalTokensBytes[bytes] {
            return [id]
        }

        var parts = bytes.map {
            [$0]
        }

        while true {
            var bestID = Int32.max
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
    
    //debug slop
    func diffGraphJSON(_ file1: String, _ file2: String) {
        func load(_ filename: String) -> [[String: Any]]? {
            guard let url = Bundle.main.url(forResource: filename, withExtension: nil) else {
                print("File not found:", filename)
                return nil
            }
            do {
                let data = try Data(contentsOf: url)
                return try JSONSerialization.jsonObject(with: data) as? [[String: Any]]
            } catch {
                print(error)
                return nil
            }
        }

        guard let a = load(file1), let b = load(file2) else { return }

        func extract(_ items: [[String: Any]]) -> (
            buffers: [Int: [String: Any]],
            bufferData: [Int: String],
            programs: [String: String],
            programUsage: Set<String>,
            bufferUsage: Set<Int>
        ) {
            var buffers = [Int: [String: Any]]()
            var bufferData = [Int: String]()
            var programs = [String: String]()
            var programUsage = Set<String>()
            var bufferUsage = Set<Int>()

            for item in items {
                guard let key = item.keys.first else { continue }

                switch key {
                case "buff_alloc":
                    if let info = item[key] as? [String: Any],
                       let num = info["num"] as? Int {
                        buffers[num] = info
                    }

                case "copyin":
                    if let info = item[key] as? [String: Any],
                       let dest = info["dest"] as? Int,
                       let data = info["data"] as? String {
                        bufferData[dest] = data
                    }

                case "program":
                    if let info = item[key] as? [String: Any],
                       let name = info["name"] as? String,
                       let lib = info["lib"] as? String {
                        programs[name] = lib
                    }

                case "call":
                    if let info = item[key] as? [String: Any] {
                        if let name = info["name"] as? String {
                            programUsage.insert(name)
                        }
                        if let bufs = info["buffers"] as? [Int] {
                            for b in bufs {
                                bufferUsage.insert(b)
                            }
                        }
                    }

                default:
                    break
                }
            }

            return (buffers, bufferData, programs, programUsage, bufferUsage)
        }

        let x = extract(a)
        let y = extract(b)

        print("\n=== Programs only in \(file1) ===")
        for p in Set(x.programs.keys).subtracting(y.programs.keys) {
            print(p)
        }

        print("\n=== Programs only in \(file2) ===")
        for p in Set(y.programs.keys).subtracting(x.programs.keys) {
            print(p)
        }

        print("\n=== Buffers only in \(file1) ===")
        for b in Set(x.buffers.keys).subtracting(y.buffers.keys) {
            print(b)
        }

        print("\n=== Buffers only in \(file2) ===")
        for b in Set(y.buffers.keys).subtracting(x.buffers.keys) {
            print(b)
        }

        print("\n=== Buffers with different copyin data ===")
        for b in Set(x.bufferData.keys).intersection(y.bufferData.keys) {
            if x.bufferData[b] != y.bufferData[b] {
                print("buffer \(b)")
            }
        }

        print("\n=== Programs with different binaries ===")
        for p in Set(x.programs.keys).intersection(y.programs.keys) {
            if x.programs[p] != y.programs[p] {
                print(p)
            }
        }

        print("\n=== Programs used only in one ===")
        for p in x.programUsage.subtracting(y.programUsage) {
            print("\(p) only used in \(file1)")
        }
        for p in y.programUsage.subtracting(x.programUsage) {
            print("\(p) only used in \(file2)")
        }

        print("\n=== Buffers used only in one ===")
        for b in x.bufferUsage.subtracting(y.bufferUsage) {
            print("\(b) only used in \(file1)")
        }
        for b in y.bufferUsage.subtracting(x.bufferUsage) {
            print("\(b) only used in \(file2)")
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
            //let graph = GraphRunner(filename: "1.rc")
            //graph.diffGraphJSON("1.rc", "1b.rc")
            //print("1")
            //run_tests()
            
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
    encode_graph = GraphRunner(filename: "0.rc")
    var ref_wav = load_audio(file: file, samplingRate: 24000)
    let wav_len = ref_wav.count
    ref_wav = expandWav(ref_wav)
    memcpy(buffers[encode_graph.copyins.last!]!.contents(), ref_wav, ref_wav.count * MemoryLayout<Float>.stride)
    encode_graph.run()
    var ref_audio_tokens = get_ref_tokens()
    model_graph = GraphRunner(filename: "1.rc")
    for b in encode_graph.buffs.subtracting(model_graph.buffs) { buffers[b] = nil }
    ref_audio_tokens = ref_audio_tokens .map { Array($0.prefix(wav_len / CHUNK_SIZE)) }
    var styleTokens = tokenizer.encode("<|denoise|><|lang_start|>\(language)<|lang_end|><|instruct_start|>None<|instruct_end|>")
    let chunks = getChunks(text: text, refText: refText, wavLen: wav_len, styleTokens: styleTokens, num_ref_tokens: Int(wav_len / CHUNK_SIZE))
    var rets: [[[Int32]]] = []
    for chunk in chunks {
        var tokens: [[Int32]] = Array(repeating: Array(repeating: Int32(AUDIO_MASK_ID), count: MAX_LEN), count: NUM_AUDIO_CODEBOOK)
        let target_length = estimateTargetTokens(text: chunk, refText: refText, numRefAudioTokens: ref_audio_tokens[0].count)
        let (sched, num_steps) = get_sched(numSteps: num_steps, targetLength: target_length)
        let combined = [refText, chunk].map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }.joined(separator: " ")
        let text_tokens = tokenizer.encode("<|text_start|>\(combined)<|text_end|>").map { Int32($0) }
        var (c_len, audio_mask, attention_mask, input_ids) = getInputs(textTokens: text_tokens, targetLength: target_length, refAudioTokens: ref_audio_tokens, styleTokens: styleTokens)
        for step in 0..<num_steps {
            //copyins
            
            let input_ids_flat = input_ids.flatMap { $0.flatMap { $0 } }
            buffers[1135]!.contents().copyMemory(from: input_ids_flat, byteCount: input_ids_flat.count * MemoryLayout<Int32>.stride)
            
            let attention_mask_flat = attention_mask.flatMap { $0.flatMap { $0.flatMap { $0 } } }
            buffers[1134]!.contents().copyMemory(from: attention_mask_flat, byteCount: attention_mask_flat.count)
            
            let tokens_flat = tokens.flatMap { $0 }
            buffers[1136]!.contents().copyMemory(from: tokens_flat, byteCount: tokens_flat.count * MemoryLayout<Int32>.stride)
            
            let audio_mask_flat = audio_mask.flatMap { $0 }
            buffers[1080]!.contents().copyMemory(from: audio_mask_flat, byteCount: audio_mask_flat.count)
            
            model_graph.run(vals_dict: [131: target_length ,367: c_len])
            let scores_out = Array(UnsafeBufferPointer(start: buffers[model_graph.copyouts[0]]!.contents().assumingMemoryBound(to: Float32.self), count: buffer_sz[model_graph.copyouts[0]]!))[0..<(NUM_AUDIO_CODEBOOK * MAX_LEN)]
            let n = scores_out.count / NUM_AUDIO_CODEBOOK
            var scores = stride(from: 0, to: scores_out.count, by: n).map { Array(scores_out[$0..<min($0 + n, scores_out.count)])}
            
            let pred_tokens_out = Array(UnsafeBufferPointer(start: buffers[1704]!.contents().assumingMemoryBound(to: Float32.self), count: buffer_sz[1704]!))[0..<(MAX_LEN * NUM_AUDIO_CODEBOOK)]
            var pred_tokens = (0..<NUM_AUDIO_CODEBOOK).map { i in Array(pred_tokens_out[(i * MAX_LEN)..<((i + 1) * MAX_LEN)])}
            
            
            print(step,"scores =",scores)
            scores = scores.map { Array($0.prefix(target_length)) }
            pred_tokens = pred_tokens.map { Array($0.prefix(target_length)) }
                
            let flatScores = scores.flatMap { $0 }
            print("\n\n",flatScores, flatScores.count)
            let sortedIdx = flatScores.indices.sorted { flatScores[$0] > flatScores[$1]}
            //print(sortedIdx)
            let topkIdx = Array(sortedIdx.prefix(sched[step]))
            
            //todo untested from here
            
            var sampleTokens = tokens.map { Array($0.prefix(target_length)) }
            sampleTokens = sampleTokens.flatMap { $0 }
            
            let predFlat = pred_tokens.flatMap { $0 }
            var sampleTokensFlat = sampleTokens.flatMap { $0 }
            for (i, idx) in topkIdx.enumerated() { sampleTokensFlat[idx] = Int32(predFlat[idx]) }
            
            sampleTokens = stride(from: 0, to: sampleTokensFlat.count, by: target_length).map { Array(sampleTokensFlat[$0..<($0 + target_length)]) }
            
            for i in 0..<NUM_AUDIO_CODEBOOK { for j in 0..<target_length { tokens[i][j] = sampleTokens[i][j] } }
            print(sampleTokens)
            
            for i in 0..<NUM_AUDIO_CODEBOOK {
                for j in 0..<target_length {
                    input_ids[0][i][c_len - target_length + j] = sampleTokens[i][j]
                    input_ids[1][i][j] = sampleTokens[i][j]
                }
            }
            
            print(model_graph.copyins)
        }
        rets.append(tokens)
    }
    print("rory rets =",rets)
    print("1")
}


func get_sched(numSteps: Int, targetLength: Int) -> ([Int], Int) {
    let timesteps = (0...numSteps).map { i -> Double in
        Double(i) / Double(numSteps)
    }.map { t -> Double in
        (T_SHIFT * t) / (1 + (T_SHIFT - 1) * t)
    }

    let totalMask = targetLength * NUM_AUDIO_CODEBOOK
    var rem = totalMask
    var sched: [Int] = []

    for step in 0..<numSteps {
        let num: Int

        if step == numSteps - 1 {
            num = rem
        } else {
            num = min(
                Int(ceil(Double(totalMask) * (timesteps[step + 1] - timesteps[step]))),
                rem
            )
        }
        sched.append(num)
        if num >= MAX_LEN {
            return get_sched(
                numSteps: numSteps * 2,
                targetLength: targetLength
            )
        }
        rem -= num
    }
    return (sched, numSteps)
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

func getInputs(textTokens: [Int32], targetLength: Int, refAudioTokens: [[Int32]], styleTokens: [Int32]) -> (Int, [[Bool]], [[[[Bool]]]], [[[Int32]]]) {
    let targetAudioTokens = Array(repeating: Int32(AUDIO_MASK_ID), count: targetLength)
    let c_len = styleTokens.count + textTokens.count + refAudioTokens[0].count + targetLength
    let condAudioStartIdx = c_len - targetLength - refAudioTokens[0].count

    var condinput_ids: [[[Int32]]] = [[]]
    for i in 0..<NUM_AUDIO_CODEBOOK { condinput_ids[0].append(styleTokens + textTokens + refAudioTokens[i] + targetAudioTokens) }
    var input_ids = Array(
        repeating: Array(
            repeating: Array(repeating: Int32(AUDIO_MASK_ID), count: MAX_LEN),
            count: NUM_AUDIO_CODEBOOK
        ),
        count: 2
    )

    for i in 0..<NUM_AUDIO_CODEBOOK {for j in 0..<c_len {input_ids[0][i][j] = condinput_ids[0][i][j]}}
    for i in 0..<NUM_AUDIO_CODEBOOK {
        let start = condinput_ids[0][i].count - targetLength
        for j in 0..<targetLength { input_ids[1][i][j] = condinput_ids[0][i][start + j] }
    }

    let condaudio_mask = Array(repeating: false, count: condAudioStartIdx) + Array(repeating: true, count: c_len - condAudioStartIdx)

    var audio_mask = Array(repeating: Array(repeating: false, count: MAX_LEN), count: 2)
    for i in 0..<c_len {audio_mask[0][i] = condaudio_mask[i]}
    for i in 0..<targetLength { audio_mask[1][i] = condaudio_mask[condaudio_mask.count - targetLength + i]}

    var attentionMask = Array(
        repeating: Array(
            repeating: Array(
                repeating: Array(repeating: false, count: MAX_LEN),
                count: MAX_LEN
            ),
            count: 1
        ),
        count: 2
    )
    for i in 0..<c_len { for j in 0..<c_len { attentionMask[0][0][i][j] = true } }
    for i in 0..<targetLength { for j in 0..<targetLength { attentionMask[1][0][i][j] = true }}
    for i in targetLength..<c_len { attentionMask[1][0][i][i] = true}

    return (c_len, audio_mask, attentionMask, input_ids)
}

func expandWav(_ refWav: [Float], ref_audio_length: Int = REF_AUDIO_LEN) -> [Float] {
    var refWav = refWav
    let clipSize = refWav.count % CHUNK_SIZE

    if clipSize > 0 {
        refWav = Array(refWav.dropLast(clipSize))
    }

    let wavLen = refWav.count
    let targetLen = SAMPLING_RATE * ref_audio_length

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

func get_ref_tokens(ref_audio_length: Int = REF_AUDIO_LEN) -> [[Int32]] {
    let count = buffer_sz[encode_graph.copyouts[0]]!
    let flat = Array(UnsafeBufferPointer(start: buffers[encode_graph.copyouts[0]]!.contents().assumingMemoryBound(to: Int32.self), count: count))
    let prefixCount = Int((REF_AUDIO_LEN * SAMPLING_RATE * NUM_AUDIO_CODEBOOK) / CHUNK_SIZE)
    let trimmed = Array(flat.prefix(prefixCount))
    let n = trimmed.count / NUM_AUDIO_CODEBOOK
    return stride(from: 0, to: trimmed.count, by: n).map { Array(trimmed[$0..<($0 + n)]) }
}

func getChunks(text: String, refText: String, wavLen: Int, styleTokens: [Int32], num_ref_tokens: Int) -> [String] {
    print("rory inputs")
    print(text)
    print(refText)
    print(wavLen)
    print(styleTokens)
    print(num_ref_tokens)
    let pattern = #"[^。，！？；：、.,?]+[。，！？；：、.,?]?"#
    let regex = try! NSRegularExpression(pattern: pattern, options: [])

    let nsText = text as NSString
    let matches = regex.matches(in: text, options: [], range: NSRange(location: 0, length: nsText.length))

    var chunksSmall: [String] = matches.map {
        nsText.substring(with: $0.range)
    }

    var chunks: [String] = [""]
    var j = 0

    for i in 0..<chunksSmall.count {
        if chunksSmall[i].first == " " {
            chunksSmall[i].removeFirst()
        }

        let combined = chunks[j] + chunksSmall[i]
        
        let targetLength = estimateLargestTargetTokens(text: combined, refText: refText, numRefAudioTokens: Int(wavLen / CHUNK_SIZE))

        let joinedText = [refText, combined].map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }.joined(separator: " ")

        let textTokens = tokenizer.encode("<|text_start|>\(joinedText)<|text_end|>")

        if styleTokens.count + textTokens.count + num_ref_tokens + targetLength < MAX_LEN {

            chunks[j] += chunksSmall[i]
        } else {
            chunks.append(chunksSmall[i])
            j += 1
        }
    }

    return chunks
}

func estimateLargestTargetTokens(text: String, refText: String, numRefAudioTokens: Int) -> Int {
    let refWeight = 2.5 * Double(refText.count)
    let speedFactor = refWeight / Double(numRefAudioTokens)
    let maxCharWeight = Double(CHAR_WEIGHTS.max() ?? 0.0)
    let targetWeight = maxCharWeight * Double(text.count)
    let estimatedDuration = targetWeight / speedFactor
    return Int(estimatedDuration)
}

func estimateTargetTokens(text: String, refText: String, numRefAudioTokens: Int,) -> Int {
    func weightSum(for string: String) -> Double {return string.unicodeScalars.reduce(0.0) { sum, scalar in sum + Double(CHAR_WEIGHTS[Int(scalar.value)] ?? 0.0) }}
    let refWeight = weightSum(for: refText)
    let speedFactor = refWeight / Double(numRefAudioTokens)
    let targetWeight = weightSum(for: text)
    let estimatedDuration = targetWeight / speedFactor
    return Int(estimatedDuration)
}

//todo......
func run_tests() {
    //audio load
    var value = load_audio(file: "voice3_short")
    var expected = (try! JSONDecoder().decode([Float].self, from: Data(contentsOf: Bundle.main.url(forResource: "voice3_ref_wav", withExtension: "json")!)))
    assert(value.count == expected.count && zip(value, expected).allSatisfy { abs($0 - $1) < 1e-5 })
    value = load_audio(file: "voice4_short")
    expected = (try! JSONDecoder().decode([Float].self, from: Data(contentsOf: Bundle.main.url(forResource: "voice4_ref_wav", withExtension: "json")!)))
    assert(value.count == expected.count && zip(value, expected).allSatisfy { abs($0 - $1) < 1e-5 })
    
    // expand to REF_AUDIO_LEN s
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
    
    
    value = (try! JSONDecoder().decode([Float].self, from: Data(contentsOf: Bundle.main.url(forResource: "voice3_ref_wav_exp", withExtension: "json")!)))
    memcpy(buffers[encode_graph.copyins.last!]!.contents(), value, value.count * MemoryLayout<Float>.stride)
    encode_graph.run()
    out = get_ref_tokens()
    expected_tokens = try! JSONDecoder().decode([[[Int32]]].self, from: Data(contentsOf: Bundle.main.url(forResource: "voice3_ref_audio_tokens", withExtension: "json")!))[0]
    assert(out == expected_tokens, "Token mismatch: got \(out), expected \(expected_tokens)")
    
    //tokenizer test
    
    let language = "None"
    let tok = Tokenizer()
    var toks = tokenizer.encode("<|denoise|><|lang_start|>\(language)<|lang_end|><|instruct_start|>None<|instruct_end|>")
    var ref_tokens = try! JSONDecoder().decode([[[Int32]]].self, from: Data(contentsOf: Bundle.main.url(forResource: "voice4_ref_audio_tokens", withExtension: "json")!))[0]
    assert(toks == [151669, 151670, 4064, 151671, 151672, 4064, 151673])
    toks = tokenizer.encode("<|text_start|>That's it, turn the page on the day, walk away 'Cause there's sense in what I say, I'm forty-fifth generation roman but I don't know them or care when I'm spitting, So return to your sitting position and listen, it's fitting that I'm miles ahead and they chase me, show your face on TV then we'll see, you can't do half My crew laughs at your rhubarb-and-custard verses You rain down curses, but I'm waving your hearses driving by Streets riding high with the beats in the sky All stare, eyes glazed, garage burnt down The fire raged for forty days and in forty ways But through the blaze, they see it fade The sea of black.<|text_end|>")
    assert(toks == [151674, 4792, 594, 432, 11, 2484, 279, 2150, 389, 279, 1899, 11, 4227, 3123, 364, 60912, 1052, 594, 5530, 304, 1128, 358, 1977, 11, 358, 2776, 35398, 2220, 57610, 9471, 47776, 714, 358, 1513, 944, 1414, 1105, 476, 2453, 979, 358, 2776, 978, 14810, 11, 2055, 470, 311, 697, 11699, 2309, 323, 8844, 11, 432, 594, 26345, 429, 358, 2776, 8756, 8305, 323, 807, 32486, 752, 11, 1473, 697, 3579, 389, 5883, 1221, 582, 3278, 1490, 11, 498, 646, 944, 653, 4279, 3017, 13627, 48236, 518, 697, 21669, 44497, 65, 9777, 1786, 590, 567, 49299, 1446, 11174, 1495, 67147, 11, 714, 358, 2776, 63111, 697, 52059, 288, 9842, 553, 65518, 19837, 1550, 448, 279, 33327, 304, 279, 12884, 2009, 45843, 11, 6414, 92186, 11, 19277, 49340, 1495, 576, 3940, 435, 3279, 369, 35398, 2849, 323, 304, 35398, 5510, 1988, 1526, 279, 62473, 11, 807, 1490, 432, 15016, 576, 9396, 315, 3691, 13, 151675])
    
    //test chunks
    value = try! JSONDecoder().decode([Float32].self, from: Data(contentsOf: Bundle.main.url(forResource: "voice4_ref_wav", withExtension: "json")!))
    var text = "Testing testing one two three, this is made with Omni-Voice. Can you hear me? or not? thank you for listening to this"
    let ref_text = "This is a wav file for my voice, so that omni voice can capture my voice. I need to talk for about 15 seconds"
    let wav_len = 171139
    var chunks = ["Testing testing one two three,this is made with Omni-Voice.Can you hear me?or not?", "thank you for listening to this"]
    toks = tokenizer.encode("<|denoise|><|lang_start|>\(language)<|lang_end|><|instruct_start|>None<|instruct_end|>")
    var chunks_out = getChunks(text: text, refText: ref_text, wavLen: wav_len, styleTokens: toks, num_ref_tokens: Int(value.count / CHUNK_SIZE))
    assert(chunks_out == chunks, "mismatch: got \(chunks_out), expected \(chunks)")
    
    text = "That's it, turn the page on the day, walk away 'Cause there's sense in what I say, I'm forty-fifth generation roman but I don't know them or care when I'm spitting, So return to your sitting position and listen, it's fitting that I'm miles ahead and they chase me, show your face on TV then we'll see, you can't do half My crew laughs at your rhubarb-and-custard verses You rain down curses, but I'm waving your hearses driving by Streets riding high with the beats in the sky All stare, eyes glazed, garage burnt down The fire raged for forty days and in forty ways But through the blaze, they see it fade The sea of black."
    chunks = ["That's it,turn the page on the day,walk away 'Cause there's sense in what I say,", "I'm forty-fifth generation roman but I don't know them or care when I'm spitting,", "So return to your sitting position and listen,it's fitting that I'm miles ahead and they chase me,", "show your face on TV then we'll see,", "you can't do half My crew laughs at your rhubarb-and-custard verses You rain down curses,", "but I'm waving your hearses driving by Streets riding high with the beats in the sky All stare,eyes glazed,", "garage burnt down The fire raged for forty days and in forty ways But through the blaze,", "they see it fade The sea of black."]
    chunks_out = getChunks(text: text, refText: ref_text, wavLen: wav_len, styleTokens: toks, num_ref_tokens: Int(value.count / CHUNK_SIZE))
    assert(chunks_out == chunks, "mismatch: got \(chunks_out), expected \(chunks)")
    
    
    //get inputs
    var tokens: [Int32] = [151674, 1986, 374, 264, 53807, 1034, 369, 847, 7743, 11, 773, 429, 7861, 7751, 7743, 646, 12322, 847, 7743, 13, 358, 1184, 311, 3061, 369, 911, 220, 16, 20, 6486, 26768, 7497, 825, 1378, 2326, 22416, 374, 1865, 448, 85225, 19625, 8834, 53280, 498, 6723, 752, 30, 269, 537, 30, 151675]
    let styleTokens: [Int32] = [151669, 151670, 4064, 151671, 151672, 4064, 151673].map { Int32($0) }
    var ref_audio_tokens = try! JSONDecoder().decode([[[Int32]]].self, from: Data(contentsOf: Bundle.main.url(forResource: "voice4_ref_audio_tokens", withExtension: "json")!))[0]
    ref_audio_tokens = ref_audio_tokens .map { Array($0.prefix(wav_len / CHUNK_SIZE)) }
    let (c_len, audio_mask, attention_mask, input_ids) = getInputs(
        textTokens: tokens,
        targetLength: 131,
        refAudioTokens: ref_audio_tokens,
        styleTokens: styleTokens
    )
    let target_length = estimateTargetTokens(text: "Testing testing one two three,this is made with Omni-Voice.Can you hear me?or not?", refText: "This is a wav file for my voice, so that omni voice can capture my voice. I need to talk for about 15 seconds", numRefAudioTokens: ref_audio_tokens[0].count)
    assert(target_length == 131)
    let exp_audio_mask = try! JSONDecoder().decode([[Bool]].self, from: Data(contentsOf: Bundle.main.url(forResource: "exp_audio_mask", withExtension: "json")!))
    let exp_attention_mask = try! JSONDecoder().decode([[[[Bool]]]].self, from: Data(contentsOf: Bundle.main.url(forResource: "exp_attention_mask", withExtension: "json")!))
    let exp_input_ids = try! JSONDecoder().decode([[[Int32]]].self, from: Data(contentsOf: Bundle.main.url(forResource: "exp_input_ids", withExtension: "json")!))
    assert(c_len == 367)
    assert(exp_audio_mask == audio_mask)
    assert(exp_attention_mask == attention_mask)
    assert(exp_input_ids == input_ids)
    
    let (sched, num_steps) = get_sched(numSteps: 16, targetLength: 131)
    assert(sched == [7, 8, 9, 11, 12, 14, 17, 20, 25, 31, 40, 53, 75, 115, 198, 413])
    assert(num_steps == 16)
    
    model_graph = GraphRunner(filename: "1.rc")
    for b in encode_graph.buffs.subtracting(model_graph.buffs) { buffers[b] = nil }
    //model_graph.run()

    print("DONE")
}


