# ORPDAD
 This is the official code repository of our dataset and ECCV 2024 paper entitled "Oulu Remote-photoplethysmography Physical Domain Attacks Database (OPDAD)". 

In this study we propose the first dataset containing a wide set of physical domain attack scenarios divided in three categories (illumination, movement, concealment) that directly target the main weaknesses of rPPG. As rPPG can be easily influenced by the recording environment, this vulnerability can therefore be exploited to inject fake signals or impair predictions physically. ORPDAD was collected as a new benchmark dataset to evaluate the robustness of rPPG to Physical Domain Attacks. In our paper we analyse a total of 13 rPPG methods (6 hand-crafted + 7 deep learning) and their susceptibility to the proposed attack scenarios. We hope that our dataset can encourage further research in security to physical attacks and that it will aid the development of new more robust methods. 

You can check our or paper on the following link (COMING SOON). In this repository you will find all the necessary code to run baseline methods on the pre-processed data, and also the pre-processing code. Due to strict regulations governing the use of sensitive data we cannot publish it directly, if you want to utilize our dataset for academic purposes please contact guoying.zhao@oulu.fi to request a copy.

# Physical Domain Attacks in RPPG

As rPPG is an emerging technology that deals with sensitive data and could be employed in applications where security is critical (clinical, remote driver monitoring, anti-spoofing, authentication, \etc), it is crucial to critically assess its vulnerabilities early and work towards mitigating potential security exploits. This dataset focuses on the scarcely studied physical-domain attacks. A physical-domain attack assumes that the attacker cannot influence the digital data after acquisition or the rPPG method used for estimation, namely the attacker can only inject noise during the acquisition process using physical means. We study three major vulnerabilities and attack scenario categories. Firstly, as rPPG is measured from reflected light, a possible attack vector is the light source. Adding a weak perturbation signal to the incident light source means that it will be present in the estimated rPPG signal as well. Secondly, subject movement induces noise to the extracted signal, consequently having the subject move at a set rhythm will make this added motion perturbation strongly periodical. Therefore, we can leverage both motion and illumination to inject strongly periodical (rPPG like) signals into the data. Thirdly, the light measured is reflected from the skin, so either blocking the light (with an opaque cloth) or attenuating it (with make-up) can affect the rPPG negatively. Consequently we define these three physical-domain attack  categories: **Illumination (I)**, **Movement (M)** and **Concealment (C)**.

# What does ORPDAD contain

We record data from **30 participants** ( with no history of heart disease) under different scenarios based on the three aforementioned categories of attacks. We record **70 second** videos for each scenario, for a total of 26 scenarios per subject, resulting in 780 videos equivalent to around 15.2h of footage. For the synchronised ground truth physiological signals we record both PPG waveforms via a finger-oximeter and ECG waveforms via an ECG device. Based on the defined attack categories, we define the following scenarios:

* Still (S1-S3): Subject is recorded at a still and resting state before each attack category. Motion is kept at a comfortable minimum.

* Illumination (I1-I6): We combine the two attack frequencies (50bpm, 100bpm) and three different intensity settings (S=7\%, M=14\% and L=21\% of the attack device max brightness), resulting in I1=50S, I2=50M, I3=50L, I4=100S, I5=100M, I6=100L. The intensities are much weaker than the base lighting, and cause perturbations are visually imperceptible in the recordings.

* Movement (M1-M11): We explore simple movements such as small vertical (SV), large vertical (LV), small horizontal (SH), large horizontal (LH), mouth open/close (M) at the two target frequencies (50bpm, 100bpm). Resulting in M1=50SV, M2=50LV, M3=50SH, M4=50LH,  M5=50M, M6=100SV, M7=100LV, M8=100SH, M9=100LH,  M10=100M. We also include natural talking motion M11 that is not at a set attack frequency.

* Concealment (C1-C6): The first three opaque (C1-C3) scenarios involve wearing a forehead concealing beanie C1, a facial mask C2 and then both C3. The transparent concealment scenarios (C4-C6) contain progressive application of make-up with primer C4, foundation C5 and setting powder C6.

![ATTACKVIS](visualisation_of_example_attack_scenarios.png)
