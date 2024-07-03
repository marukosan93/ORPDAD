# ORPDAD
 This is the official code repository of our dataset and ECCV 2024 paper entitled "Oulu Remote-photoplethysmography Physical Domain Attacks Database (OPDAD)". 

In this study we propose the first dataset containing a wide set of physical domain attack scenarios divided in three categories (illumination, movement, concealment) that directly target the main weaknesses of rPPG. As rPPG can be easily influenced by the recording environment, this vulnerability can therefore be exploited to inject fake signals or impair predictions physically. ORPDAD was collected as a new benchmark dataset to evaluate the robustness of rPPG to Physical Domain Attacks. In our paper we analyse a total of 13 rPPG methods (6 hand-crafted + 7 deep learning) and their susceptibility to the proposed attack scenarios. We hope that our dataset can encourage further research in security to physical attacks and that it will aid the development of new more robust methods. 

You can check our or paper on the following link (COMING SOON). In this repository you will find all the necessary code to run baseline methods on the pre-processed data, and also the pre-processing code. Due to strict regulations governing the use of sensitive data we cannot publish it directly, if you want to utilize our dataset for academic purposes please contact guoying.zhao@oulu.fi to request a copy.

# Physical Domain Attacks in RPPG

As rPPG is an emerging technology that deals with sensitive data and could be employed in applications where security is critical (clinical, remote driver monitoring, anti-spoofing, authentication, \etc), it is crucial to critically assess its vulnerabilities early and work towards mitigating potential security exploits. This dataset focuses on the scarcely studied physical-domain attacks. A physical-domain attack assumes that the attacker cannot influence the digital data after acquisition or the rPPG method used for estimation, namely the attacker can only inject noise during the acquisition process using physical means. We study three major vulnerabilities and attack scenario categories. Firstly, as rPPG is measured from reflected light, a possible attack vector is the light source. Adding a weak perturbation signal to the incident light source means that it will be present in the estimated rPPG signal as well. Secondly, subject movement induces noise to the extracted signal, consequently having the subject move at a set rhythm will make this added motion perturbation strongly periodical. Therefore, we can leverage both motion and illumination to inject strongly periodical (rPPG like) signals into the data. Thirdly, the light measured is reflected from the skin, so either blocking the light (with an opaque cloth) or attenuating it (with make-up) can affect the rPPG negatively. Consequently we define these three physical-domain attack  categories: **Illumination (I)**, **Movement (M)** and **Concealment (C)**.

![VULNEVIS](rppg_physical_vulnerabilities.png)
# What does ORPDAD contain

We record data from **30 participants** ( with no history of heart disease) under different scenarios based on the three aforementioned categories of attacks. We record **70 second** videos for each scenario, for a total of **26 scenarios** per subject, resulting in **780 videos** equivalent to ≈**15.2h** of footage. For the synchronised ground truth physiological signals we record both PPG waveforms via a finger-oximeter and ECG waveforms via an ECG device. Based on the defined attack categories, we define the following scenarios:

* **Still (S1-S3)**: Subject is recorded at a still and resting state before each attack category. Motion is kept at a comfortable minimum.

* **Illumination (I1-I6)**: We combine the two attack frequencies (50bpm, 100bpm) and three different intensity settings (S=7\%, M=14\% and L=21\% of the attack device max brightness), resulting in I1=50S, I2=50M, I3=50L, I4=100S, I5=100M, I6=100L. The intensities are much weaker than the base lighting, and cause perturbations are visually imperceptible in the recordings.

* **Movement (M1-M11)**: We explore simple movements such as small vertical (SV), large vertical (LV), small horizontal (SH), large horizontal (LH), mouth open/close (M) at the two target frequencies (50bpm, 100bpm). Resulting in M1=50SV, M2=50LV, M3=50SH, M4=50LH,  M5=50M, M6=100SV, M7=100LV, M8=100SH, M9=100LH,  M10=100M. We also include natural talking motion M11 that is not at a set attack frequency.

* **Concealment (C1-C6)**: The first three opaque (C1-C3) scenarios involve wearing a forehead concealing beanie C1, a facial mask C2 and then both C3. The transparent concealment scenarios (C4-C6) contain progressive application of make-up with primer C4, foundation C5 and setting powder C6.

![ATTACKVIS](visualisation_of_example_attack_scenarios.png)

For each recording we provide:
- 1920x1080p 30fps RGB video
- synchronised groundtruth from both PPG and ECG devices
- 128x128 cropped videos centered at the face (for end-to-end methods)
- 68 point facial landmarks for each frame
- spatial-temporal map representations for each video (for non-end-to-end methods)

# How ORPDAD was collected
To study each attack vector in an isolated manner, we setup a controlled recording environment as to minimise the influence of external factors that are not related to the attack. Thus, we keep the same base environment for all subjects and scenarios, enabling us to study the specific contribution induced by the attacks. For the base video recordings we use a professional RGB camera and two fixed led lights. We record the physiological signals with a finger PPG oximeter and a portable ECG belt device. All the recording devices are synchronised by keeping internal clock times consistent. The subjects are seated at ≈1m from the camera and the lights are positioned above the camera and illuminate the subject at a 45° angle at ≈1.5m distance. 

For the illumination attacks we design two modulated light panels that each operate at 1600 lm maximum, by using led strips and a control circuit. These panels allow the injected light signal to be more uniformly distributed over the face surface instead of attacking only part of it (single led case). We pilot the panels with Pulse Width Modulation (PWM) signals, thus being able to create a light source that can output any waveform and at any fraction of the max intensity. For this experiment we simulate a typical rPPG waveform, with the main frequency component placed at the target frequency and a second harmonic at 1/4 of the peak intensity. For the L, M, H settings we use 7%, 14%, 21% of the max power and set the lights at ≈1m distance from the subject. For the movement scenarios we simply utilise an audio metronome and instruct the subjects to move according to the set rhythm and movement type. For opaque concealment we use medical face masks and black beanie hats. For translucent concealment we choose a primer, foundation and setting powder, and keep the brand and shade (220 natural beige) consistent among all subjects. 


![VULNERABILITIESVIS](collection_setup.png)


# Dataset Structure


```
ORPDAD_RGB_videos_partX.tar.gz #divided in X = 1,..., 10 parts for smaller size download files
├── 0XX # participants ID goes from 001 to 030
│   └── video
│       ├──  SCEN.mov # ≈70s(≈2100frames) RGB video at 1920x1080 30fps
│       └──  ..... # for each of the 26 SCEN -> S1-S3, I1-I6, M1-M11, C1-C6
ORPDAD_DATA.tar.gz #contains all the processed data and ground truth (cropped videos, bvp and ecg, spatial-temporal maps, landmarks )
├── 0XX # participants ID goes from 001 to 030
│   ├── blockh5
│   │   ├──  SCEN_block128.h5 # SCEN_block128.h5["video"] is a  HDF5 for the array containing the cropped video [T≈2100, H=128, W=128, C=3]
│   │   └──  ..... # for each of the 26 SCEN -> S1-S3, I1-I6, M1-M11, C1-C6
│   ├── maps
│   │   ├──  SCEN_mstmap.npy
│   │   ├──  SCEN_stmap.npy
│   │   └──  ..... # for each of the 26 SCEN -> S1-S3, I1-I6, M1-M11, C1-C6
│   ├── ecg
│   │   ├──  SCEN_ecg.npy
│   │   ├──  SCEN_pseudobvp.npy
│   │   └──  ..... # for each of the 26 SCEN -> S1-S3, I1-I6, M1-M11, C1-C6
│   ├── bvp
│   │   ├──  SCEN_bvp.npy
│   │   ├──  SCEN_hr.npy
│   │   └──  ..... # for each of the 26 SCEN -> S1-S3, I1-I6, M1-M11, C1-C6
│   └── lnd
│       ├──  SCEN_lnd.npy
│       └──  ..... # for each of the 26 SCEN -> S1-S3, I1-I6, M1-M11, C1-C6
```

# Evaluation Protocols 

# Training/Evaluation code

# Pre-processing code
