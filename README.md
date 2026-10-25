# Autocomplete · Автопродолжение

Letter-level autocomplete that runs entirely in your browser. As you type, the rest of the word appears in grey; after a space,
it predicts the next word. Press Tab to accept, ↑ ↓ to cycle through options, or tap the suggestions below the input field.

## How it works

- **Model.** A hierarchical letter-level model (letters in, letters out) without a token vocabulary. Characters are embedded into
  word vectors, processed by linear and softmax attention blocks, and passed to a letter decoder to predict the next character.
- **In the browser.** Six ONNX graphs (letter encoder/decoder, word reader, decoder condition, copy head, and body) run via
  onnxruntime-web (WebAssembly) inside a Web Worker. The body steps through one word at a time, carrying over its state.
- **Suggestions.** Generated using beam search over characters up to the end of the word. A copy head integrates exact matches
  from the text you've already typed.

## Limits

- Trained on a mix of 20 languages, but quality was evaluated exclusively on Russian web text.
- The initial load requires a ~75 MB download. Subsequent visits skip the download (browser cache); the model still takes a few seconds to start.
- All generated suggestions are displayed; there is no confidence threshold yet.

## Privacy

No backend, no analytics. All processing happens locally, and your text never leaves the browser.
