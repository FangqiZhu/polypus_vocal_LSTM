Case study of LSTM for polypus detection based on preprocessing features from vocal data

vocal data:
The recording of the voice for automatic diagnosis of the problem is achieved based on the speech analysis. We collect the data that are about the discrete vocal samples of recording 12 respondents, which 4 of them have throat polypus and the rest are healthy (no throat polypus). In the study, the vowels /a:/ and /i:/ are collected and each of them lasts for approximate 10 seconds. The samples of vowels within the same scenarios are snipped off a longer vowel sample to choose the stable part of the whole sample for processing.

Remark: 
1. Scipy and MATLAB use different FFT libraries. Scipy uses lapack, while MATLAB uses FFTW. 
These libraries use different algorithms and produce slightly different results.
2. The dimension of the S matrix so absolute different, please keep consensus for usage.
3. The loading part will load the Spectrogram matrix extracted from MATLAB function spectrogram.m
