from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

m_lengths = {1:31, 2:29, 3: 31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
year_list = []

for month, ndays in m_lengths.items():
    for day in range(ndays):
        year_list.append([day + 1, month])

dt_list = [datetime.strptime(f'{dt[0]}/{dt[1]}/2024', r"%d/%m/%Y").isocalendar()[1:] for dt in year_list]

def WHT(predicted_values, real_values, verbose=False):
  p_fft = np.fft.fft(predicted_values)
  r_fft = np.fft.fft(real_values)

  plt.figure(figsize=(16, 4))
  plt.title("Reconstructed Composite Waveform")
  plt.xlabel("Time")
  plt.ylabel("Amplitude")

  waves = []

  for fft_result in [r_fft, p_fft]:

    magnitude = np.abs(fft_result)
    phase = np.angle(fft_result)
    # Number of frequency bins
    num_bins = len(fft_result)

    # Frequency values corresponding to each bin
    # This depends on the sampling rate and the length of the input signal
    # For simplicity, we assume a unit sampling rate.
    frequency_values = np.fft.fftfreq(num_bins)
    if verbose:
      plt.figure(figsize=(12, 6))
      plt.subplot(2, 1, 1)
      plt.stem(frequency_values, magnitude)
      plt.title("Magnitude Spectrum")
      plt.xlabel("Frequency (Hz)")
      plt.ylabel("Magnitude")

      # Plot the phase
      plt.subplot(2, 1, 2)
      plt.stem(frequency_values, phase)
      plt.title("Phase Spectrum")
      plt.xlabel("Frequency (Hz)")
      plt.ylabel("Phase (radians)")

      plt.tight_layout()
      plt.show()
      plt.close()
    
    avg_mag = np.mean(magnitude)
    significant_freqs = [i for i, mag in enumerate(magnitude) if mag > avg_mag]
    time = np.arange(num_bins)
    individual_waves = []
    
    for frequency in significant_freqs:
      component_wave = np.cos(2 * np.pi * frequency_values[frequency] * time)
      individual_waves.append(component_wave)
    
    # Combine and plot the reconstructed composite waveform
    reconstructed_waveform = np.sum(individual_waves, axis=0)
    waves.append(reconstructed_waveform)

    plt.plot(time, reconstructed_waveform)

  plt.show()
  
  return waves