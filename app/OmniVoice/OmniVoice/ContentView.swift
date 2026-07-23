//
//  ContentView.swift
//  OmniVoice
//
//  Created by Rory Clear on 22/07/2026.
//

import SwiftUI
import Foundation

var buffers: [Int: MTLBuffer] = [:]
var buffer_sz: [Int: Int] = [:] // todo
var programs: [String: MTLLibrary] = [:]
var encode_graph: GraphRunner!

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
                            programs[name] = library
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
        let device = MTLCreateSystemDefaultDevice()!
        let queue = device.makeCommandQueue()!

        for item in self.calls {
            let name = item["name"] as! String
            let library = programs[name]!
            let function = library.makeFunction(name: name)!
            
            let pipeline = try! device.makeComputePipelineState(function: function)

            let commandBuffer = queue.makeCommandBuffer()!
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
            encode_graph = GraphRunner(filename: "encode.rc")
            runGenerate()
            let data = Data(bytes: buffers[encode_graph.copyouts[0]]!.contents(), count: buffer_sz[encode_graph.copyouts[0]]!)
            let first1000Bytes = data.prefix(1000)

            let intCount = first1000Bytes.count / MemoryLayout<Int32>.size

            let intArray: [Int32] = first1000Bytes.withUnsafeBytes { ptr in
                let buffer = ptr.bindMemory(to: Int32.self)
                return Array(buffer.prefix(intCount))
            }

            print(intArray)
            
        }
    }

    func runGenerate() {
        let url = Bundle.main.url(forResource: "voice4", withExtension: "wav")!
        let audioData = try! Data(contentsOf: url)
        generate(
            text: "Random text",
            refText: "Reference text",
            refAudio: audioData,
            numSteps: 16,
            language: "en"
        )
    }
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
    
    print("ref_wav length:", ref_wav.count)
    var sum = ref_wav.reduce(0.0) { $0 + Double($1) }
    print("ref_wav sum:", sum)
    
    let clipSize = ref_wav.count % chunk_size
    if clipSize > 0 { ref_wav.removeLast(clipSize) }
    let wavLen = ref_wav.count
    let targetLength = sampling_rate * 20
    if wavLen < targetLength { ref_wav.append(contentsOf: Array(repeating: 0.0, count: targetLength - wavLen)) }
    
    print("\nref_wav length:", ref_wav.count)
    sum = ref_wav.reduce(0.0) { $0 + Double($1) }
    print("ref_wav sum:", sum)
        
    ref_wav.withUnsafeBytes { memcpy(buffers[encode_graph.copyins[encode_graph.copyins.count - 1]]!.contents(), $0.baseAddress!, ref_wav.count * MemoryLayout<Float>.size) }
    
    //print(encode_graph.copyins)
    encode_graph.run()
    //print(encode_graph.copyouts)
    
    //print("length:", ref_wav.count)

    //print("first 1000 values:")
    //print(Array(ref_wav.prefix(1000)))

    //let sum = ref_wav.reduce(0.0) { $0 + Double($1) }
    //print("sum:", sum)
    
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

