In this project, I have decided to apply lightweight mobilenetv2 and heavyweight vgg19 on blood smear images of NIH malaria dataset to compare their performances and detect malaria by classifying 'Parasitized' cell images against 'Uninfected' cell images.

Malaria is a life-threatening, mosquito-borne infectious disease caused by single-celled Plasmodium parasites and transmitted by the bites of infected female Anopheles mosquitoes. 
Once injected into the bloodstream, the parasites travel to the liver to mature and multiply, then re-enter the blood to infect and rupture red blood cells. It mainly attacks the liver and destroys red blood cells.
Rarely, transmission can occur via blood transfusions, organ transplants, or from a mother to an unborn child. It does not spread directly from person to person through casual contact.
Symptoms can include fever, chills, headaches, muscle aches, nausea, vomiting,fatigue,confusion, seizures, severe anemia, jaundice, respiratory distress, kidney failure, coma,etc.
Diagnosis can be confirmed by microscopic examination of blood smears or rapid diagnostic tests (RDTs) that detect malaria antigens. 
Most people diagnosed with malaria recover completely when it is diagnosed and treated early with the right antimalarial medications. Severe cases need emergency hospitalization and intravenous or injectable medications.

The NIH malaria dataset consists of 13779 uninfected blood smear cells & 13779 uninfected blood smear cells. 

<img width="558" height="393" alt="image" src="https://github.com/user-attachments/assets/e2215a62-a002-4150-9792-bdcfca099664" />

All input images are resized to a standard width and height of 124 pixels as neural networks need uniform input shapes to process images as fixed-size tensors.
32 images are fed to the model at one time before updating the internal weights as this batch size is a standard default that balances training speed with memory efficiency.

The images are normalized and resized to 124×124 & randomly augments images from a directory for binary classification. 
It automatically splits the dataset to 80% shuffled batches for training and 20% ordered batches for validation by sharing one generator applies random augmentations to validation data as well.

27,558 images are organized in 2 folders (classes) & are cleanly split to 80% for training (22,048 images) and 20% for validation (5,510 images).
Everything loaded properly without missing files or extra hidden folders.

Now we will be applying and building the VGG19 & Mobilenetv2 architectures for classification.
VGG-19 is a deep convolutional neural network architecture proposed from the Visual Geometry Group (VGG) at the University of Oxford with 19 weight layers (16 convolutional layers & 3 fully connected layers).

A transfer learning model is built by a frozen VGG19 base (pretrained on ImageNet without its original top layers) to extract features from input images.
It then attaches a custom classification head having a Flatten layer, a 256-unit Dense layer, 50% Dropout & a single sigmoid output unit—to predict binary malaria infection probabilities.

A lightweight transfer learning model with a frozen MobileNetV2 base is created that retains its pre-learned visual features without retraining its core layers. 
It then adds a compact classification head—using global average pooling, a 128-unit dense layer, 30% dropout, and a single final output unit—to check if a cell image shows malaria infection.

3-channel input tensor shape (124, 124, 3) for RGB images is set for the training duration to 10 full passes through the dataset.
VGG19 transfer learning model and the Mobilenetv2 transfer learning model are set with the 124x124x3 input shape & compiles it for training. 
It configures the Adam optimizer with a conservative learning rate of 0.0001. 
Binary cross-entropy as the loss function was used for two-class prediction & tracks accuracy during training. 
VGG19 and Mobilenetv2 are trained on the training images for 10 full epochs while checking the accuracy on the validation images after each cycle.
Both the base models are configured for 224×224 color images and sets up an early stopping rule to halt training if the validation error fails to improve for two consecutive cycles while automatically keeping the best-performing model weights. 
It then trains the existing VGG19 model for upto 12 full cycles on the training data, checking its progress against the validation data after each cycle and applying that early stopping safeguard.
The training line still runs on the previously built 124×124 model.

The classification metrics generated after the epochs are as follows:

Generating predictions on validation dataset...
173/173 ━━━━━━━━━━━━━━━━━━━━ 33s 187ms/step
173/173 ━━━━━━━━━━━━━━━━━━━━ 36s 185ms/step

============================================================
 1. MODEL COMPACTNESS (NUMBER OF PARAMETERS)
============================================================
VGG19 Total Parameters:       21,204,545
MobileNetV2 Total Parameters: 2,422,081
Reduction Factor:             MobileNetV2 is 8.8x smaller!

============================================================
 2. DETAILED CLASSIFICATION REPORTS
============================================================

[VGG19 CLASSIFICATION REPORT]
              precision    recall  f1-score   support

 Parasitized       0.95      0.85      0.90      2755
  Uninfected       0.86      0.96      0.91      2755

    accuracy                           0.90      5510
   macro avg       0.91      0.90      0.90      5510
weighted avg       0.91      0.90      0.90      5510


[MOBILENETV2 CLASSIFICATION REPORT]
              precision    recall  f1-score   support

 Parasitized       0.94      0.91      0.93      2755
  Uninfected       0.92      0.94      0.93      2755

    accuracy                           0.93      5510
   macro avg       0.93      0.93      0.93      5510
weighted avg       0.93      0.93      0.93      5510


This output compares the size and validation performance of both models across all 5,510 validation images (2,755 parasitized and 2,755 uninfected), showing that MobileNetV2 outperforms VGG19 in both accuracy and efficiency.

Inference from here:

a) VGG19 uses 21,204,545 parameters while Mobilenetv2 uses 2,422,081 parameters. This makes Mobilenetv2 almost 8.75 times smaller than VGG19.
b) MobileNetV2 gets 93% overall accuracy, beating VGG19's 90% accuracy on the exact same validation dataset.
c) MobileNetV2 catches 91% of actual parasitized cells compared to 85% for VGG19. In simple words Mobilenetv2 misses far fewer true malaria infections.
d) MobileNetV2 maintains an F1-score of 0.93 across both infected and uninfected cells, delivering better and more consistent predictions while using a fraction of the memory.

<img width="1359" height="490" alt="image" src="https://github.com/user-attachments/assets/17ae5c5d-452e-4343-9c38-8fa2fee770d7" />

From the confusion matrices of the 2 models, we can infer that:

a) MobileNetV2 detects 179 more infected cells than VGG19
b) MobileNetV2 cuts missed infections by 42.6% 
c) VGG19 triggers 43 fewer false alarms on healthy cells
d) Both models clear approximately 95% of healthy cells accurately

MobileNetV2 drops missed infections from 420 down to 241 that makes it a better model than VGG19 as in medical imaging, missing a sick patient is far more dangerous than a false alarm.
Across all 5,510 validation images, MobileNetV2 makes 397 total mistakes (241 + 156) compared to 533 total mistakes (420 + 113) for VGG19.

<img width="1390" height="490" alt="image" src="https://github.com/user-attachments/assets/0e4fe8de-c315-4be9-accc-33a5defde0cd" />

From the above learning curves, we infer that: 
a) MobileNetV2 learns faster, gets higher accuracy & maintains much lower error rates than VGG19 across all 10 epochs.
b) MobileNetV2 starts its first cycle at roughly 91.4% validation accuracy and 0.23 loss, already surpassing VGG19's best performance at the end of training ( almost 91.0% accuracy).
c) MobileNetV2's validation accuracy climbs smoothly to a peak of 93.3% around cycle 8, while its validation loss drops consistently down to almost 0.186.
d) VGG19 starts lower at almost 87.0% accuracy and 0.31 loss, showing noticeable fluctuations like the loss spike at epoch 8 as it struggles to catch up.
e) Both models show downward-trending loss curves through the final cycle, meaning neither model has begun memorizing the training data yet.

<img width="1590" height="990" alt="image" src="https://github.com/user-attachments/assets/5d1e15c3-8e60-49f3-a121-e585b970bfd3" />

Frome these training-versus-validation curves we see that:
a) Both models generalize cleanly without severe overfitting
b) MobileNetV2 gets a noticeably higher performance ceiling across both training and validation sets.
c) In both VGG19 (top row) and MobileNetV2 (bottom row), the dashed validation lines closely follow the solid training lines. 
d) Neither models show the classic sign of overfitting, where training error drops while validation error climbs sharply upward.
e) Training & validation curves of VGG19 stay nearly on top of each other, ending at around 91% accuracy and 0.22–0.23 loss.
f) The frozen VGG19 base is likely underfitting slightly and lacks the capacity to extract finer malaria cell features without unfreezing deeper layers as the training accuracy plateaus near 91%.
g) MobileNetV2 climbs quickly from 89% to around 94.2% training accuracy, pulling its validation accuracy up to almost 93.3% with only a small, healthy gap(almost 1%) between training & validation performance.
h) Both models still show slight downward trends in validation loss at the end of epoch 9, meaning a few extra training cycles or light fine-tuning has the chance to improve MobileNetV2's accuracy even higher.

<img width="702" height="547" alt="image" src="https://github.com/user-attachments/assets/a77a6014-0ed5-4da5-b99e-de6e99093e18" />

The ROC-AUC curves confirm the following things:
a) Both models strongly distinguish between infected and uninfected cells.
b) MobileNetV2 gets a slightly higher Area Under the Curve (AUC = 0.9782) than VGG19 (AUC = 0.9700).
c) Both curves hug the top-left corner far above the dashed diagonal line (which represents random guessing at an AUC of 0.50) showing excellent separability between parasitized & uninfected blood cells.
d) In the crucial left-hand area where the False Positive Rate is below 0.10, MobileNetV2's orange curve rises more steeply than VGG19's blue curve showing • MobileNetV2 catches more true infections without triggering unnecessary false alarms.


Evaluating inference speed over 5510 images...

==================================================
             INFERENCE TIME COMPARISON            
==================================================
VGG19:
  Total Inference Time: 34.87 seconds
  Average Per-Image:    6.3276 ms

MobileNetV2:
  Total Inference Time: 29.50 seconds
  Average Per-Image:    5.3534 ms
--------------------------------------------------
Speedup: MobileNetV2 is 1.18x faster than VGG19!
==================================================

Now this benchmark shows that:
a) MobileNetV2 is almost 15.4% faster than VGG19, processing all 5,510 validation images in 29.50 seconds compared to VGG19's 34.87 seconds.
b) MobileNetV2 averages 5.35 ms per image (with almost 187 images per second), saving roughly 0.97 ms per cell over VGG19's 6.33 ms per image (almost 158 images per second).
c) Processing nearly 190 cell crops per second allows MobileNetV2 to scan an entire blood smear slide with thousands of cells in just a few seconds.

Now a final a 1000-iteration bootstrap statistical stability test will be done on the validation predictions of both models to prove that MobileNetV2's higher ROC-AUC score is statistically reliable rather than a random fluke.
It will repeatedly draw 1000 random samples (with replacement) from the 5510 validation predictions & recalculates the ROC-AUC score for both models on every sample.

============================================================
            ROC-AUC VARIANCE & STABILITY REPORT            
============================================================
VGG19 ROC-AUC:
  Mean AUC:            0.97001
  Variance:            3.92e-06
  Standard Deviation:  0.00198
  95% Confidence Int:  [0.96605, 0.97379]

MobileNetV2 ROC-AUC:
  Mean AUC:            0.97822
  Variance:            2.58e-06
  Standard Deviation:  0.00161
  95% Confidence Int:  [0.97501, 0.98144]
============================================================

This report shows that: 
a) MobileNetV2’s performance advantage is statistically significant and more consistent than VGG19’s, as their 95% confidence intervals do not overlap.
b) Mean ROC-AUC of MobileNetV2 leads by +0.00821 AUC to the mean ROC-AUC of VGG19
c) Zero overlap at 95% confidence interval confirms a real, non-random improvement
d) MobileNetV2’s lowest 95% confidence bound (0.97501) is still higher than VGG19’s upper bound (0.97379) ruling out random sampling luck across the 5510 validation images.
e) MobileNetV2 exhibits 34% less variance than VGG19 (2.58e-6 vs 3.92e-6), showing that its diagnostic accuracy holds steady across varied subsets of patient cell images.

<img width="855" height="547" alt="image" src="https://github.com/user-attachments/assets/2ff5b4f6-5599-44cf-a5d5-4381af432cbd" />

This histogram proves that: 
a) MobileNetV2 is both significantly more accurate and noticeably more consistent than VGG19 across all 1,000 bootstrap samples.
b) The orange MobileNetV2 distribution sits cleanly to the right of the blue VGG19 distribution, centered at 0.9782 vs 0.97, with only tiny overlap in the extreme tails at around 0.974–0.975.
c) MobileNetV2 forms a tighter, taller bell curve with roughly 34% lower variance (2.58e-6 compared to VGG19's 3.92e-6).
d) The tighter spread shows stronger statistical stability across different subsets of cell images.

From the project, we conclude that: 
a) MobileNetV2 is significantly superior to VGG19 for automated malaria parasite detection in blood smear images. 
b) While VGG19 suffers from a heavy parameter footprint and higher false-negative rates, MobileNetV2 achieves 93% accuracy, a 0.9782 ROC-AUC, and a 42.6% reduction in missed malaria infections—all while being 8.75 times smaller & 15.4% faster. 
c) Combined with its statistically proven stability, MobileNetV2 gives an accurate, lightweight & reliable deep learning solution suitable for real-time clinical screening & deployment on low-resource diagnostic hardware.
d) Bigger models aren't always better—especially in healthcare.
e) MobileNetV2 is nearly 8.75 times smaller than VGG19, yet it achieved gets accuracy & cut missed malaria infections by over 42%. 
f) Since the results are validated with a 1,000-iteration bootstrap test showing zero overlap in confidence intervals, we know this improvement is statistically reliable. 
g) Combined with its 5-millisecond prediction speed, MobileNetV2 is ready to deploy on affordable mobile devices in clinics that need fast malaria screening most.





