# Electrophysiology-Chemical-Drug-Signal-Identification

This repository aims to develop a system for detecting and classifying chemicals based on the electromagnetic signatures emitted from the human body when consuming them. The hypothesis is that different chemicals have the opportunity to make the body generate unique electromagnetic waves, which can be captured using an antenna and then classified to identify the substance. The goal is to help society with the information that can be gathered in order to help us understand the negative impacts of drugs. I see this being used for chemical research, chemical detection, chemical overdose prevention, and to find out the physiological and psychological stance of a subject (if they are angry, sad, afraid, happy, etc). We have to do what we can in order to prevent a serious rise in drug epademics around the world. Today drugs are being made in labs with the purpose of hooking subjects a lot faster than before. It is advisable to learn whatever we can in order to help society evolve and not degrade from the use of harmful substances.

**PREVENT THE USE OF CHEMICALS FOR MALICIOUS PURPOSES AT ALL COSTS, CHEMICAL WARFARE IS REAL, AND A DANGER TO ALL**

## Table of contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Model](#model)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Usage](#usage)
8. [Contributing](#contributing)
9. [License](#license)

## Introduction

Drug abuse is a global issue, leading to severe health problems and affecting communities in various ways. It is crucial to develop tools that can detect and identify drug consumption. This repository presents an approach using electromagnetic signatures released by the human body when consuming drugs. The goal is to classify these signals to identify the specific substance involved.

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/electrophysiology-drug-signal-identification.git
cd electrophysiology-drug-signal-identification
```
2. **Install the required packages:**
```bash
pip install -r requirements.txt
```

## Dataset

The dataset should consist of electromagnetic signals collected from individuals who have consumed various drugs. It should include a diverse set of substances, demographic factors, and recording conditions. Each data point should have a corresponding label indicating the drug consumed.

**To prepare the dataset, follow these steps:**

1. Collect raw electromagnetic signals using an antenna (done when the subject is under the influence of the drug). Antenna must be tuned to a certain frequency band to capture the electromagnetism, please reach out to me to find out more about this. You are able to recieve these signals from a far distance, you do not even have to be close to the subject...
2. Preprocess the data, removing unwanted noise and normalizing signal amplitude. I used a FASTDTW to make training data out of the audio data I collected from the signals. -> as done with sound_cutter_fastdtw.py
3. Split the dataset into training, validation, and test sets -> done by split_data.py. Currenty it is Training set size: 136, Validation set size: 29, Test set size: 30 <- this will change as we populate it with more data.
4. Determine data formats -> Binary File Sink (straight from source), WAV, MCSS, etc. Currently I have only done it for WAV and MCSS, but I have worked and am working on soucing data from the source (using an RTL SDR).
5. Save the data as `.npy` or `.csv` files in the `data/npy_files` directory -> done by save_npy.py.

## Models
**NOTE: TBD still working on this.**
Listing models we currently ran it on.
- Convolution Neural Network (CNN)

## Training
**NOTE: TBD still working on this.**
To train the model, run the following command:
```bash
python train.py --epochs EPOCHS --batch_size BATCH_SIZE --lr LEARNING_RATE
```
where `EPOCHS`, `BATCH_SIZE`, and `LEARNING_RATE` are the desired number of epochs, batch size, and learning rate, respectively.

## Evaluation
**NOTE: TBD still working on this.**
To evaluate the trained model, run the following command:
```bash
python evaluate.py --model_path PATH_TO_MODEL
```

The script will output the classification accuracy, confusion matrix, and other relevant metrics.

## Usage
**NOTE: TBD still working on this.**
To classify a new electromagnetic signal, run the following command:
```bash
python predict.py --model_path PATH_TO_MODEL --signal_path PATH_TO_SIGNAL
```

The script will output the predicted drug class for the input signal.

## Contributing

Pull requests are welcome. For significant changes, please open an issue first to discuss the proposed change.

## Science

The release of electromagnetic radiation from the human body can be primarily attributed to thermal radiation and bioelectric activity. Here, I provide some basic equations related to these phenomena:

**Thermal Radiation:**
The human body emits thermal radiation due to its temperature. This radiation follows the Stefan-Boltzmann law:
![CodeCogsEqn](https://user-images.githubusercontent.com/102178068/236707223-8c161cf2-9ee0-498c-8b52-59a03688ac78.png)

where:
- P: Power emitted as thermal radiation (W)
- σ: Stefan-Boltzmann constant (5.67 × 10^-8 W m^-2 K^-4)
- A: Surface area of the body (m^2)
- T: Temperature of the body in Kelvin (K)

**Bioelectric Activity:**
Bioelectric phenomena occur due to the movement of charged particles, such as ions, within cells and tissues. The bioelectric potential, V, can be described by the cable equation:

![CodeCogsEqn(1)](https://user-images.githubusercontent.com/102178068/236707232-737c9f14-8fa4-4335-ba68-cdead08fb256.png)


where:
- V: Membrane potential (V)
- t: Time (s)
- λ: Electrotonic length constant (m)
- x: Distance along the cable (m)
- E: Equilibrium potential (V)

**The electrotonic length constant, λ, is given by:**

![CodeCogsEqn(2)](https://user-images.githubusercontent.com/102178068/236707248-6107bdfd-406d-4ae2-8373-9dae709a35a2.png)

where:
- R_m: Membrane resistance per unit length (Ω m)
- R_i: Intracellular resistance per unit length (Ω m)
Please note that these equations are general and not specific to drug-induced electromagnetic signatures. Analyzing such signatures requires capturing the electromagnetic signals from the human body and then applying signal processing and machine learning techniques to classify the drug type.

Please note that these equations are general and not specific to drug-induced electromagnetic signatures. Analyzing such signatures requires capturing the electromagnetic signals from the human body and then applying signal processing and machine learning techniques to classify the drug type which we are attempting do to. I would find it super interesting to see some equations that relate to drug consumption and the electrophysiology of the human body...

## License

This project is licensed under the ____ - see the [LICENSE](LICENSE) file for details...
