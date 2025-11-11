# BSL-Static-48: A Dataset of Static Bangla Sign Alphabets and Digits

## Description

This dataset provides a collection of images and extracted landmark features for 48 fundamental static signs in Bangla Sign Language (BSL), including 38 alphabets and 10 digits (0-9). It aims to support research in isolated sign language recognition (SLR) for BSL, addressing the need for accessible and processed datasets. The dataset includes raw images captured from multiple participants, mirrored versions for data augmentation, and pre-processed 126-dimensional hand landmark feature vectors, ready for use in machine learning models.

**Note:** Facial features in the raw and mirrored images have been anonymized (blurred) to protect participant privacy before public release.

---

## Folder Structure

The dataset is organized as follows:

📁 BSL_Static_48_Dataset/
│
├── 📜 README.md                     <-- এই ফাইল (ডেটাসেটের বিবরণী)
│
├── 📁 01_Images/                     <-- সমস্ত ছবির মূল ফোল্ডার
│   │
│   ├── 📁 Raw_Images/             <-- আসল অ্যানোনিমাইজ করা ছবি (.jpg)
│   │   ├── 📁 D0/
│   │   ├── 📁 D1/
│   │   │   └── ... (ছবি)
│   │   ├── ... (এভাবে D9 পর্যন্ত)
│   │   ├── 📁 L1/
│   │   ├── 📁 L2/
│   │   │   └── ... (ছবি)
│   │   └── ... (এভাবে L38 পর্যন্ত)
│   │
│   └── 📁 Mirrored_Images/        <-- মিরর করা অ্যানোনিমাইজ ছবি (.jpg)
│       ├── 📁 D0/
│       ├── 📁 D1/
│       │   └── ... (ছবি)
│       ├── ... (এভাবে D9 পর্যন্ত)
│       ├── 📁 L1/
│       ├── 📁 L2/
│       │   └── ... (ছবি)
│       └── ... (এভাবে L38 পর্যন্ত)
│
└── 📁 02_Processed_Features_NPY/    <-- প্রসেস করা ফিচার (.npy), বেঞ্চমার্কিংয়ের জন্য বিভক্ত
    │
    ├── 📁 train/                 <-- ট্রেইনিং ডেটা (৮০%)
    │   ├── 📁 D0/
    │   │   └── ... (.npy ফাইল)
    │   ├── 📁 D1/
    │   │   └── ... (.npy ফাইল)
    │   ├── ... (এভাবে D9 পর্যন্ত)
    │   ├── 📁 L1/
    │   │   └── ... (.npy ফাইল)
    │   ├── 📁 L2/
    │   │   └── ... (.npy ফাইল)
    │   └── ... (এভাবে L38 পর্যন্ত)
    │
    ├── 📁 val/                   <-- ভ্যালিডেশন ডেটা (১০%)
    │   ├── 📁 D0/
    │   │   └── ... (.npy ফাইল)
    │   ├── ... (D1 থেকে L38)
    │
    └── 📁 test/                  <-- টেস্ট ডেটা (১০%)
        ├── 📁 D0/
        │   └── ... (.npy ফাইল)
        ├── ... (D1 থেকে L38)





---

## Class Mapping

[cite_start]The class labels used in the folder names correspond to the following Bangla signs [cite: 819-822]:

**Digits (D0-D9):**
* `D0`: ০
* `D1`: ১
* `D2`: ২
* `D3`: ৩
* `D4`: ৪
* `D5`: ৫
* `D6`: ৬
* `D7`: ৭
* `D8`: ৮
* `D9`: ৯

**Alphabets (L1-L38):**
* `L1`: অ/য়
* `L2`: আ
* `L3`: ই/ঈ
* `L4`: উ/ঊ
* `L5`: ঋ/র/ড়/ঢ়
* `L6`: এ
* `L7`: ঐ
* `L8`: ও
* `L9`: ঔ
* `L10`: ক
* `L11`: খ/ক্ষ
* `L12`: গ
* `L13`: ঘ
* `L14`: ঙ
* `L15`: চ
* `L16`: ছ
* `L17`: জ/য
* `L18`: ঝ
* `L19`: ঞ
* `L20`: ট
* `L21`: ঠ
* `L22`: ড
* `L23`: ঢ
* `L24`: ণ/ন
* `L25`: ত/ৎ
* `L26`: থ
* `L27`: দ
* `L28`: ধ
* `L29`: প
* `L30`: ফ
* `L31`: ব/ভ
* `L32`: ম
* `L33`: ল
* `L34`: শ/ষ/স
* `L35`: হ
* `L36`: ং (*Anusvara*)
* `L37`: ঃ (*Visarga*)
* `L38`: ঁ (*Chandrabindu*)

---

## Data Collection

* **Source:** Images were collected from **5 volunteers** (friends).
* **Equipment:** A **Macbook Air M3 camera** was used.
* **Environment:** Data was captured indoors under **room lighting** conditions against a **white background**.
* **Method:** Images were captured manually using a Python script (`static_data_collector.py`) by pressing the 'SPACE' key to ensure clear frames.
* **Anonymization:** Facial regions in all images within `01_Images/` were blurred using OpenCV's DNN face detector and Gaussian blurring (`anonymize_faces.py` script) prior to public release to ensure participant privacy, as formal informed consent for sharing identifiable images was not obtained.

---

## Feature Extraction

* **Framework:** **MediaPipe Holistic** (check your installed version via `pip show mediapipe`) was used via the `static_feature_extractor.py` script.
* **Features:** For each image (raw and mirrored), 126 features were extracted, corresponding to the 3D coordinates (x, y, z) of the 21 landmarks for the **left hand** (63 features) and the **right hand** (63 features). Features were filled with zeros if a hand was not detected. Pose and face landmarks were *not* included in the processed feature files.
* **Format:** Features are saved as NumPy arrays in `.npy` files in the `02_Processed_Features_NPY/` directory.

---

## Data Splitting

* The `.npy` feature files were split into `train`, `val`, and `test` sets using the `static_data_splitter.py` script with an 80%/10%/10% ratio, stratified by class. The raw/mirrored images in `01_Images/` are not pre-split.

---

## Dataset Statistics

### Dataset Statistics

[cite_start]This dataset contains a total of **29132** processed feature samples across **48** static Bangla Sign Language classes[cite: 823].
[cite_start]The data is organized into raw images, mirrored images, and processed NumPy feature files (126 features per sample)[cite: 824].

**Class Distribution:**

| Class Label | Raw Images | Mirrored Images | Total Features | Train Features | Val Features | Test Features |
| :---------- | :--------- | :-------------- | :------------- | :------------- | :----------- | :------------ |
| D0          | 302        | 302             | 604            | 483            | 60           | 61            |
| D1          | 297        | 297             | 594            | 475            | 59           | 60            |
| D2          | 300        | 300             | 600            | 480            | 60           | 60            |
| D3          | 300        | 300             | 600            | 480            | 60           | 60            |
| D4          | 301        | 301             | 602            | 481            | 60           | 61            |
| D5          | 300        | 300             | 600            | 480            | 60           | 60            |
| D6          | 300        | 300             | 600            | 480            | 60           | 60            |
| D7          | 301        | 301             | 602            | 481            | 60           | 61            |
| D8          | 300        | 300             | 600            | 480            | 60           | 60            |
| D9          | 301        | 301             | 602            | 481            | 60           | 61            |
| L1          | 300        | 300             | 600            | 480            | 60           | 60            |
| L10         | 325        | 325             | 650            | 520            | 65           | 65            |
| L11         | 305        | 305             | 610            | 488            | 61           | 61            |
| L12         | 305        | 305             | 610            | 488            | 61           | 61            |
| L13         | 305        | 305             | 610            | 488            | 61           | 61            |
| L14         | 304        | 304             | 608            | 486            | 61           | 61            |
| L15         | 305        | 305             | 610            | 488            | 61           | 61            |
| L16         | 305        | 305             | 610            | 488            | 61           | 61            |
| L17         | 305        | 305             | 610            | 488            | 61           | 61            |
| L18         | 305        | 305             | 610            | 488            | 61           | 61            |
| L19         | 304        | 304             | 608            | 486            | 61           | 61            |
| L2          | 301        | 301             | 602            | 481            | 60           | 61            |
| L20         | 304        | 304             | 608            | 486            | 61           | 61            |
| L21         | 305        | 305             | 610            | 488            | 61           | 61            |
| L22         | 305        | 305             | 610            | 488            | 61           | 61            |
| L23         | 303        | 303             | 606            | 484            | 61           | 61            |
| L24         | 305        | 305             | 610            | 488            | 61           | 61            |
| L25         | 305        | 305             | 610            | 488            | 61           | 61            |
| L26         | 317        | 317             | 634            | 507            | 63           | 64            |
| L27         | 305        | 305             | 610            | 488            | 61           | 61            |
| L28         | 306        | 306             | 612            | 489            | 61           | 62            |
| L29         | 305        | 305             | 610            | 488            | 61           | 61            |
| L3          | 301        | 301             | 602            | 481            | 60           | 61            |
| L30         | 305        | 305             | 610            | 488            | 61           | 61            |
| L31         | 304        | 304             | 608            | 486            | 61           | 61            |
| L32         | 293        | 293             | 586            | 468            | 59           | 59            |
| L33         | 305        | 305             | 610            | 488            | 61           | 61            |
| L34         | 306        | 306             | 612            | 489            | 61           | 62            |
| L35         | 306        | 306             | 612            | 489            | 61           | 62            |
| L36         | 305        | 305             | 610            | 488            | 61           | 61            |
| L37         | 302        | 302             | 604            | 483            | 60           | 61            |
| L38         | 303        | 303             | 606            | 484            | 61           | 61            |
| L4          | 301        | 301             | 602            | 481            | 60           | 61            |
| L5          | 299        | 299             | 598            | 478            | 60           | 60            |
| L6          | 301        | 301             | 602            | 481            | 60           | 61            |
| L7          | 301        | 301             | 602            | 481            | 60           | 61            |
| L8          | 301        | 301             | 602            | 481            | 60           | 61            |
| L9          | 302        | 302             | 604            | 483            | 60           | 61            |
|-------------|------------|-----------------|----------------|----------------|--------------|---------------|
| **Total**   | **14566**  | **14566**       | **29132**      | **23293**      | **2911**     | **2928** |

[cite_start]*(Table data extracted from dataset_stats.txt [cite: 823-1116])*

---

## How to Cite

If you use this dataset in your research, please cite:

1.  **Our Data in Brief Article:** *(Please add citation here once the paper is published)*
    > Author(s), "Article Title," *Data in Brief*, vol. XX, p. XXXXXX, Year. DOI: XXXXX
2.  **This Dataset via Mendeley Data:** *(Please add citation here once you get the DOI from Mendeley Data)*
    > Author(s) (Year), "Dataset Title", Mendeley Data, V1, DOI: *[DOI placeholder]*

---

## License

This dataset is made available under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.

*(This license lets others distribute, remix, adapt, and build upon your work, even commercially, as long as they credit you for the original creation.)*

