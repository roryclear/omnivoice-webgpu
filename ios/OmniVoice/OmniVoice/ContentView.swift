//
//  ContentView.swift
//  OmniVoice
//
//  Created by Rory Clear on 22/07/2026.
//

import SwiftUI

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
    refAudio: Any,
    numSteps: Int = 16,
    language: String = "None"
) {
    print("generate func")
    print("audio bytes:", refAudio)
}

#Preview {
    ContentView()
}
