# Gradient Waveform Errors and How to Correct Them

Welcome! This is a code repository accompanying the paper "Gradient Waveform Errors and How to Correct Them: A review of methods to measure and correct for spatiotemporal magnetic field errors in MRI" (Harkins, K.D. et. al (In Prep.). All demos can be run as google colab notebooks, with no local dependencies.  

## Table of Contents

- [Gradient Waveform Errors and How to Correct Them](#gradient-waveform-errors-and-how-to-correct-them)
  - [Demo 0 Characterization waveform building](#demo-1-characterization-waveform-building)
  - [Demo 1 Eddy currents and correction methods](#demo-2-eddy-currents-and-correction-methods)
  - [Demo 2 GIRF measurement and correction](#demo-3-eddy-currents-and-correction-methods)
  - [Demo 3 Eddy currents and correction methods](#demo-4-eddy-currents-and-correction-methods)
  - [Contacts](#contacts)
  - [Licence](#licence)

## Demo 0: Characterization waveform building
Our first demo will start at the beginning: how do you characterize your gradient system? One of the most widely adopted methods is to input a known gradient waveform  with some frequency content of interest, and measure the output. 
This can be used to build a model of your hardware system. We will start by showing how to build a slew rate limited chirp, a typical gradient characterization waveform. 

[![Explore Siren in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jonbmartin/Gradient-Waveform-Errors-Demos/blob/main/Ex0_Characterization_Waveform_Design.ipynb
)<br>

## Demo 1: Eddy currents and correction methods
Our second demo will simulate a system with some eddy currents, which creates a nonideal (but linear) gradient system response. We will show how to correct the eddy currents. 

[![Explore Siren in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jonbmartin/Gradient-Waveform-Errors-Demos/blob/main/Ex1_Eddy_Currents.ipynb
)<br>

## Demo 2: GIRF measurement and correction
Our third demo will simulate a simple system which attenuates high frequencies (quite common). We will build a GIRF model of the system and use it to predict waveform distortions.  

[![Explore Siren in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jonbmartin/Gradient-Waveform-Errors-Demos/blob/main/Ex2_GIRF.ipynb
)<br>

## Demo 3: Neural network measurement and correction
Our fourth demo will use real data from a 7T animal scanner, and train a temporal convolutional network to predict gradient waveform errors. For this demo, make sure to set the runtime to GPU. 

[![Explore Siren in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jonbmartin/Gradient-Waveform-Errors-Demos/blob/main/Ex3_TCN.ipynb
)<br>

## Contacts

Provide a way to contact the owners of this project. It can be a team, an individual or information on the means of getting in touch via active communication channels, e.g. opening a GitHub discussion, raising an issue, etc.

## Licence

> The [LICENCE.md](./LICENCE.md) file will need to be updated with the correct year and owner

Unless stated otherwise, the codebase is released under the MIT License. This covers both the codebase and any sample code in the documentation.

Any HTML or Markdown documentation is [© Crown Copyright](https://www.nationalarchives.gov.uk/information-management/re-using-public-sector-information/uk-government-licensing-framework/crown-copyright/) and available under the terms of the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
