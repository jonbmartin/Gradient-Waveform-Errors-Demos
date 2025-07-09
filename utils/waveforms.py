import numpy as np

def chirp_slew_constrained(N, dt, G1, f1, f2, slr):
    """
    Calculates a slew-rate limited chirp waveform.

    This function is a Python conversion of the MATLAB function slrchirp.

    Reference:
    ----------
    Addy NO, Wu HH, Nishimura DG. Simple method for MR gradient system
    characterization and k-space trajectory estimation. Magn Reson Med.
    2012 Jul;68(1):120-9. doi: 10.1002/mrm.23217.

    Parameters:
    ----------
    N : int
        Number of points in the chirp.
    dt : float
        Timestep of each point (in ms).
    G1 : float
        Nominal maximum gradient amplitude (in G/cm).
    f1 : float
        Frequency at t=0 (in kHz).
    f2 : float
        Frequency at t=N*dt (in kHz).
    slr : float
        Slew rate limit (in G/cm/ms).

    Returns:
    -------
    chirp : numpy.ndarray
        The slew-rate limited chirp waveform.
    t : numpy.ndarray
        The time vector for the chirp (in ms).
    """
    # Total duration of the chirp
    T = N * dt

    # Create the time vector from 0 to (N-1)*dt
    t = np.arange(N) * dt

    # Instantaneous frequency at each time point (linear ramp)
    f = f1 + (f2 - f1) * t / T

    # Calculate the ideal chirp waveform
    # Phase of the chirp = 2*pi * integral(f(t) dt)
    phase = 2 * np.pi * (f1 * t + (f2 - f1) * t**2 / (2 * T))
    chirp = G1 * np.sin(phase)

    # Slew rate envelope of the ideal chirp
    # The slew rate is d(G)/dt, and its envelope is proportional to the
    # instantaneous frequency f.
    se = 2 * np.pi * G1 * f

    # Calculate the scaling factor based on the slew rate limit
    scaling_factor = np.minimum(slr / se, 1.0)

    # Apply the slew rate limit by scaling the chirp amplitude
    chirp = scaling_factor * chirp

    return chirp, t


def triangle_sweeps(N, dt, G1, tmin, tmax, pad_dur=0.25, negative_triangles=True,
                    moment_null=False):
    """
    Produces a series of triangle waveforms with different lengths.

    Parameters:
    ----------
    N : int
        Number of triangle durations.
    dt : float
        Timestep of each point (in ms).
    G1 : float
        Nominal maximum gradient amplitude (in G/cm).
    tmin : float
        duration of minimum-length triangle (in ms).
    tmax : float
        duration of maximum-length triangle (in ms).
    pad_dur: float
        padding between triangles (in ms).
    negative_triangles: bool
        Whether to include a negative triangle.

    Returns:
    -------
    waveform : numpy.ndarray
        The slew-rate limited chirp waveform.
    t : numpy.ndarray
        The time vector for the chirp (in ms).
    """

    tri_durs = np.linspace(tmin, tmax, N)
    pad = np.zeros(int(pad_dur//dt))
    waveform = []

    for ii in range(N):
      triangle = np.concatenate((np.linspace(0,G1,int(tri_durs[ii]//dt//2)),np.linspace(G1,0,int(tri_durs[ii]//dt//2))))
      if negative_triangles:
        waveform = np.concatenate((waveform, pad, triangle, pad, -triangle))

      else:
        waveform = np.concatenate((waveform, pad, triangle))


    waveform = np.concatenate((waveform, pad))
    waveform = waveform[:,None]

    return waveform
