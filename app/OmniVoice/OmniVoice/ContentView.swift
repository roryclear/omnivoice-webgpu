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
    let (data, sr) = loadWaveform(refAudio)
    print("sample rate:", sr)
    
    // checking
    let flatAudio = data.flatMap { $0 }

    print("first 1000 values:")
    print(Array(flatAudio.prefix(1000)))

    let sum = flatAudio.reduce(0.0) { $0 + Double($1) }
    print("sum:", sum)
}

func loadAudio(_ audio: Data, samplingRate: Int) -> ([[Float]], Int) {
    return loadWaveform(audio)
}

func loadWaveform(_ data: Data) -> ([[Float]], Int) {
    let bytes = [UInt8](data)

    // WAV header:
    // sample rate at byte offset 24 (UInt32 little endian)
    let sampleRate = Int(readUInt32LE(bytes, offset: 24))

    // channels at byte offset 22 (UInt16 little endian)
    let channels = Int(readUInt16LE(bytes, offset: 22))

    // Find "data" chunk
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

