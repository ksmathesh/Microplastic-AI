# External candidate data source

The Colab accuracy experiment can download the **training split only** of the Microplastic-Fragments-WWTP dataset directly to its temporary runtime, convert its COCO-style boxes to YOLO labels, and map the sole foreground class to the existing `Fragment` class. It does not change the original validation/test data or place these files in this repository.

- Dataset: [Microplastic-Fragments-WWTP-Dataset](https://huggingface.co/datasets/andreusv/Microplastic-Fragments-WWTP-Dataset)
- License declared by dataset card: CC BY 4.0
- Citation/DOI: https://doi.org/10.57967/hf/10280
- Source split: train only; 1,406 images (dataset card reports 910,766,614 bytes of training data)
- Domain: wastewater sludge, microscope photos at 16x–42x magnification; not necessarily visually equivalent to the project’s current samples
- Annotation mapping: source category_id 1 (microplastic fragment) -> project class `Fragment`; source category_id 0 is implicit background, with empty-label negative images preserved

The source notes that other particle morphologies can appear without annotations. Review a sample of the images and labels before relying on the supplement. Keep attribution with any use, and check the license terms for intended redistribution or publication. The source dataset itself is not bundled here.
