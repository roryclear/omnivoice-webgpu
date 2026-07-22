//
//  ContentView.swift
//  OmniVoice
//
//  Created by Rory Clear on 22/07/2026.
//

import SwiftUI
import Foundation


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
            runGenerate()
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
    print("generate func")
    print("audio bytes:", refAudio)
    let data = load_audio(refAudio, samplingRate: 24000)
    
    print("first 1000 values:")
    print(Array(data.prefix(1000)))

    let sum = data.reduce(0.0) { $0 + Double($1) }
    print("sum:", sum)
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

