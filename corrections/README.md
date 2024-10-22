### Observations

- Lines with the text "SKIP" present in the outputs with Azure OCR were not corrected because the model did not recognize these lines;
- Lines with empty text present in the outputs with Azure OCR will generate a 100% error rate because Azure OCR was unable to recognize the image.