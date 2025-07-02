This work addresses the problem of converting images of mathematical formulas into syntactically correct LaTeX markup—a task complicated by the two-dimensional nature of formulas, which limits traditional OCR methods.

The goal—to develop an approach for image-to-LaTeX conversion—was successfully achieved through:

* A review of modern methods, including symbol detection with postprocessing and end-to-end sequence generation.
* Implementation of two systems:

  * A symbol detector based on YOLO-v11 with grammar-based postprocessing.
  * A Sequence-to-Sequence model (ResNet encoder + LSTM decoder with attention).

The code is open-sourced to support reproducibility and further research. Potential applications include educational tools, digitization of scientific literature, and integration into low-resource devices.

Future directions involve hyperparameter tuning, dataset expansion (e.g., handwritten and multiline formulas), semantic context integration, model optimization (quantization/distillation), and interactive learning with human feedback.
