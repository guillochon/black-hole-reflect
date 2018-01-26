"""Calculate reflected line profiles in the galactic center."""
import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as c
from astropy import units as u
from astropy.table import Table
from PyAstronomy.modelSuite import KeplerEllipseModel
from pylab import (arccos, axis, clf, copy, cos, exp, figure, hist, rand,
                   savefig, scatter, show, sin, sqrt, subplot, transpose,
                   xlabel, ylabel)

# unit conversion: (use astropy for this?)
radians = np.pi / 180.   # deg to radians
meters = 1.0 / (((1.0 * u.meter).si.value / (
    1.0 * u.lyr / 365.25).si.value))   # light days to meters
kg = c.M_sun.si.value     # kg/solar mass
grav = c.G.si.value  # m^3/kg/s^2 gravitational constant


def get_cmap(n, name='hsv'):
    """Return a function that maps 0, 1, ..., n-1 to a distinct RGB color.

    The keyword argument name must be a standard mpl colormap name.
    """
    return plt.cm.get_cmap(name, n)


def star_position(kems, time):
    """Return the star positions for a given time."""
    # load in the star position file here:

    # calculate the position in cartesian coords:

    # random positions in a 20x20x20 box = x, y, z
    """
    num_stars = 10
    box_size = 20.
    positions = np.transpose(np.array([
        rand(num_stars) * box_size - box_size / 2.,
        rand(num_stars) * box_size - box_size / 2.,
        rand(num_stars) * box_size - box_size / 2.])) * meters
    """
    positions = np.array([
        k.evaluate(np.array([time])).tolist() for k in kems]) * 1.9e14

    # print positions
    # print positions[:,0]  # all the x-coords
    # print positions[0,:]  # x,y,z of the first star
    return positions


def star_luminosity():
    """Return the star luminosities."""
    # load in the star luminosity file here:

    luminosities = rand(40)   # random numbers for testing
    # print luminosities
    return luminosities


def rotate(x, y, co, si):
    """Rotate x, y position given cos/sin of angle."""
    xx = co * x + si * y
    yy = -si * x + co * y
    return [xx, yy]


def gas_model(num_clouds, params, other_params, plot_flag=True):
    """Retrieve the gas positions and velocities."""
    [mu, F, beta, theta_i, theta_i2, theta_o,
     kappa, mbh, f_ellip, f_flow, theta_e] = params
    [angular_sd_orbiting, radial_sd_orbiting,
        angular_sd_flowing, radial_sd_flowing] = other_params

    # First calculate the geometry of the emission:
    r = mu * F + (1. - F) * mu * beta**2. * \
        np.random.gamma(beta**(-2.), 1, num_clouds)
    phi = 2. * np.pi * rand(num_clouds)
    x = r * cos(phi)
    y = r * sin(phi)
    z = r * 0.

    # *pow(u3[i], openingBendPower));
    angle = arccos(cos(theta_o) + (1. - cos(theta_o)) * rand(num_clouds))
    cos1 = cos(angle)
    sin1 = sin(angle)
    u1 = rand(num_clouds)
    cos2 = cos(2. * np.pi * u1)
    sin2 = sin(2. * np.pi * u1)
    cos3 = cos(0.5 * np.pi - theta_i)
    sin3 = sin(0.5 * np.pi - theta_i)
    cos4 = cos(0.5 * np.pi - theta_i2)
    sin4 = sin(0.5 * np.pi - theta_i2)

    # rotate to puff up:
    [x, z] = rotate(x, z, cos1, sin1)
    # rotate to restore axisymmetry:
    [x, y] = rotate(x, y, cos2, sin2)
    # rotate to apply inclination angle:
    [x, z] = rotate(x, z, cos3, sin3)
    # rotate to apply second inclination angle:
    [x, y] = rotate(x, y, cos4, sin4)

    # weights for the different points
    # w = 0.5 + kappa * x / sqrt(x * x + y * y + z * z)
    # w /= sum(w)

    if plot_flag:
        # larger points correspond to more emission from the point
        ptsize = 5
        shade = 0.5
        clf()
        subplot(2, 2, 1)  # edge-on view 1, observer at +infinity of x-axis
        scatter(x / meters, y / meters, ptsize, alpha=shade)
        xlabel('x')
        ylabel('y')
        subplot(2, 2, 2)  # edge-on view 2, observer at +infinity of x-axis
        scatter(x / meters, z / meters, ptsize, alpha=shade)
        xlabel('x')
        ylabel('z')
        subplot(2, 2, 3)   # view of observer looking at plane of sky
        scatter(y / meters, z / meters, ptsize, alpha=shade)
        xlabel('y')
        ylabel('z')
        subplot(2, 2, 4)   # plot the radial distribution of emission
        hist(r / meters, 100)
        xlabel("r")
        ylabel("p(r)")
        show()

    # Now calculate velocities of the emitting gas:
    radius1 = sqrt(2. * grav * mbh / r)
    radius2 = sqrt(grav * mbh / r)
    vr = copy(x) * 0.
    vphi = copy(x) * 0.

    u5 = rand(num_clouds)
    n1 = np.random.normal(size=num_clouds)
    n2 = np.random.normal(size=num_clouds)
    for i in range(0, num_clouds):
        if u5[i] < f_ellip:
            # we give this point particle a near-circular orbit
            theta = 0.5 * np.pi + angular_sd_orbiting * n1[i]
            vr[i] = radius1[i] * cos(theta) * exp(radial_sd_orbiting * n2[i])
            vphi[i] = radius2[i] * sin(theta) * exp(radial_sd_orbiting * n2[i])
        else:
            if f_flow < 0.5:
                # we give this point particle an inflowing orbit
                theta = np.pi - theta_e + angular_sd_flowing * n1[i]
                vr[i] = radius1[i] * cos(theta) * \
                    exp(radial_sd_flowing * n2[i])
                vphi[i] = radius2[i] * \
                    sin(theta) * exp(radial_sd_flowing * n2[i])
            else:
                # we give this point particle an outflowing orbit
                theta = 0. + theta_e + angular_sd_flowing * n1[i]
                vr[i] = radius1[i] * cos(theta) * \
                    exp(radial_sd_flowing * n2[i])
                vphi[i] = radius2[i] * \
                    sin(theta) * exp(radial_sd_flowing * n2[i])

    # Convert vr, vphi to Cartesians:
    vx = vr * cos(phi) - vphi * sin(phi)
    vy = vr * sin(phi) + vphi * cos(phi)
    vz = vr * 0.

    # rotate to puff up:
    [vx, vz] = rotate(vx, vz, cos1, sin1)
    # rotate to restore axisymmetry:
    [vx, vy] = rotate(vx, vy, cos2, sin2)
    # rotate to apply inclination angle:
    [vx, vz] = rotate(vx, vz, cos3, sin3)
    # rotate to apply second inclination angle:
    [vx, vy] = rotate(vx, vy, cos4, sin4)

    return [x, y, z, vx, vy, vz]


def compute_gas_flux(gas_coords, star_data, times, params, plot_flag=True):
    """Calculate the flux contribution from each point particle.

    Assumptions: light travel time from stars to gas plus the
    recombination time is shorter than the time it takes the
    stars to move in their orbits.
    """
    [stellar_wind_radius, kappa] = params

    gas_flux = np.zeros((np.size(gas_coords[0]), np.size(times)))
    # load in the star luminosities (if they are constant)
    star_luminosities = star_luminosity()

    # set up the plot first
    if plot_flag:
        cmap = get_cmap(len(star_data))
        star_colors = [cmap(i) for i in range(len(star_data))]
        shade = 0.5
        min_ptsize = 1.0
        max_ptsize = 8.0
        ssize = 5.0
        boxsize = 10.0
        fig = figure(figsize=(14, 9))
        # edge-on view 1, observer at +infinity of x-axis
        axy = subplot(2, 3, 1, autoscale_on=False, aspect='equal')
        sxy = scatter([0.0], [0.0], alpha=shade,
                      edgecolors='black', linewidths=0.5)
        pxy = scatter([0.0], [0.0], c='r', s=ssize ** 2,
                      edgecolors='black', linewidths=0.5)
        axis('equal')
        xlabel('x')
        ylabel('y')
        # edge-on view 2, observer at +infinity of x-axis
        axz = subplot(2, 3, 2, autoscale_on=False, aspect='equal')
        sxz = scatter([0.0], [0.0], alpha=shade,
                      edgecolors='black', linewidths=0.5)
        pxz = scatter([0.0], [0.0], c='r', s=ssize ** 2,
                      edgecolors='black', linewidths=0.5)
        axis('equal')
        xlabel('x')
        ylabel('z')
        # view of observer looking at plane of sky
        ayz = subplot(2, 3, 3, autoscale_on=False, aspect='equal')
        syz = scatter([0.0], [0.0], alpha=shade,
                      edgecolors='black', linewidths=0.5)
        pyz = scatter([0.0], [0.0], c='r', s=ssize ** 2,
                      edgecolors='black', linewidths=0.5)
        axis('equal')
        xlabel('y')
        ylabel('z')
        avpl = subplot(2, 3, 4)   # plot the vx vs. gas flux
        vpl = scatter([0.0], [0.0], alpha=shade, s=min_ptsize,
                      edgecolors='black', linewidths=0.5)
        xlabel("v_x (10,000 km/s)")
        ylabel("Gas Flux (normalized)")
        ahpl = subplot(2, 3, 5)   # histogram of gas flux
        xlabel("v_x (10,000 km/s)")
        ylabel("Gas Flux (normalized)")

        fig.tight_layout()

    # loop over times we want spectra
    for i in range(np.size(times)):
        star_positions = star_position(star_data, times[i])
        gas_flux_values = np.zeros(
            (np.size(gas_coords[0]), np.size(star_positions)))

        # loop over the stars
        for j in range(len(star_positions)):
            r = sqrt((star_positions[j, 0] - gas_coords[0])**2. +
                     (star_positions[j, 1] - gas_coords[1])**2. +
                     (star_positions[j, 2] - gas_coords[2])**2.)
            exclude = np.zeros(len(gas_coords[0]))
            exclude[r >= stellar_wind_radius * meters] = 1.0
            # weights for the different points
            w = 0.5 + kappa * (gas_coords[0] - star_positions[j, 0]) / r
            # w /= sum(w)
            gas_flux_values[:, j] = w * exclude * \
                star_luminosities[j] / (r * r)
        gas_flux[:, i] = np.sum(gas_flux_values, axis=1)

        if plot_flag:
            # larger points correspond to more emission from the point
            gas_flux_norm = gas_flux[:, i] / sum(gas_flux[:, i])
            ptsizes = 2 * num_clouds * gas_flux_norm ** 2
            ptsizes = min_ptsize ** 2 + (max_ptsize ** 2 - min_ptsize ** 2) * (
                ptsizes - min(ptsizes)) / (max(ptsizes) - min(ptsizes))

            xy = transpose(np.array(gas_coords[:2]) / meters)
            sxy.set_sizes(ptsizes)
            sxy.set_offsets(xy)
            xy = star_positions[:, :2] / meters
            pxy.set_offsets(xy)
            pxy.set_facecolors(star_colors)
            axy.set_xlim(-boxsize, boxsize)
            axy.set_ylim(-boxsize, boxsize)

            xz = transpose(np.array(gas_coords[:3:2]) / meters)
            sxz.set_sizes(ptsizes)
            sxz.set_offsets(xz)
            xz = star_positions[:, :3:2] / meters
            pxz.set_offsets(xz)
            pxz.set_facecolors(star_colors)
            axz.set_xlim(-boxsize, boxsize)
            axz.set_ylim(-boxsize, boxsize)

            yz = transpose(np.array(gas_coords[1:3]) / meters)
            syz.set_sizes(ptsizes)
            syz.set_offsets(yz)
            yz = star_positions[:, 1:3] / meters
            pyz.set_offsets(yz)
            pyz.set_facecolors(star_colors)
            ayz.set_xlim(-boxsize, boxsize)
            ayz.set_ylim(-boxsize, boxsize)

            vx = transpose(
                np.array([gas_coords[4] / 10000000.,
                          gas_flux_norm * 1000.]))
            vpl.set_offsets(vx)
            maxx = np.max(np.abs(vx[:, 0]))
            maxy = np.max(vx[:, 1])
            avpl.set_xlim(-maxx, maxx)
            avpl.set_ylim(0, maxy)

            # No `set_data` for `hist`.
            ahpl.cla()
            hist(gas_coords[4] / 10000000., weights=gas_flux_norm,
                 bins=int(num_clouds / 100))
            ahpl.relim()
            ahpl.autoscale_view(True, True, True)
            # show()
            savefig('figs/frame-{}.png'.format(str(i).zfill(3)),
                    transparent=True, bbox_inches='tight', dpi=2 * 72)

    return gas_flux


def make_spectrum(gas_coords, gas_flux, times, wavelengths, plot_flag=True):
    """Make a spectrum for each time.

    This testing version has wavelengths = number of bins, but
    the code should eventually be passed a vector of bin centers.
    """
    # spectra = np.zeros(( np.size(times), np.size(wavelengths) ))
    spectra = np.zeros((np.size(times), int(wavelengths)))
    bin_edges = np.zeros((np.size(times), int(wavelengths) + 1))
    for i in range(0, np.size(times)):
        h = np.histogram(
            gas_coords[4], weights=gas_flux[:, i], bins=int(wavelengths))
        spectra[i, :] = h[0]
        bin_edges[i, :] = h[1]

    return [spectra, bin_edges]


def load_star_data():
    """Load stellar data."""
    kems = 1.

    gdat = Table.read('gillessen-2017.txt', format='csv')
    kems = []
    for row in gdat:
        if row['type'] != 'e' or row['period'] <= 0.0:
            continue
        kems.append(KeplerEllipseModel())
        kems[-1]['a'] = row['a']
        kems[-1]['e'] = row['e']
        kems[-1]['per'] = row['period']
        kems[-1]['tau'] = row['tperi']
        kems[-1]['w'] = row['argperi']
        kems[-1]['i'] = row['i']
        kems[-1]['Omega'] = row['long']

    return kems


########################################################################
# Set constants:
num_clouds = 10000

# Set model parameter values:
mu = 5.   # mean radius of emission, in units of light days
F = 0.5   # minimum radius of emission, in units of fraction of mu
# Gamma distribution radial profile shape parameter, between 0.01 and 2
beta = 0.5
theta_i = 10.   # x-z plane inclination angle in deg, 0 deg is face-on
theta_i2 = 60.  # x-y plane inclination angle in deg, 0 deg is face-on
# opening angle of disk in deg, 0 deg is thin disk, 90 deg is sphere
theta_o = 5.
# -0.5 emit back to center, 0 = isotropic emission, 0.5 emit away from center
kappa = -0.5
log_mbh = np.log10(4. * 10**6.)    # log10(black hole mass) in solar masses
f_ellip = 0.1  # fraction of particles in near-circular orbits
# 0-0.5 = inflow, 0.5-1 = outflow of fraction (1-f_ellip) of particles
f_flow = 0.2
# angle in plane of v_r and v_phi, 0 -> radial inflow/outflow, 90 ->
# near-circular orbits
theta_e = 20.

angular_sd_orbiting = 0.01
radial_sd_orbiting = 0.01
angular_sd_flowing = 0.01
radial_sd_flowing = 0.01

stellar_wind_radius = 0.5   # light days, radius of exclusion for line emission

params = [mu * meters, F, beta, theta_i * radians, theta_i2 * radians,
          theta_o * radians, kappa,
          kg * 10**log_mbh, f_ellip, f_flow, theta_e]  # work in SI units
params2 = [angular_sd_orbiting, radial_sd_orbiting,
           angular_sd_flowing, radial_sd_flowing]
params3 = [stellar_wind_radius, kappa]

# Set properties of predicted line profiles:
times = np.linspace(1900, 2100, 400)
wavelengths = 10   # this should be equally-spaced bins in lambda

# Load physical data:
star_data = load_star_data()

# Calculate things:
gas_coords = gas_model(num_clouds, params, params2, plot_flag=False)
gas_flux = compute_gas_flux(gas_coords, star_data,
                            times, params3, plot_flag=True)
[spectra, bin_edges] = make_spectrum(gas_coords, gas_flux, times,
                                     wavelengths, plot_flag=True)

"""
Still to do:
* add real star luminosities
* make real spectra based on specific wavelength bins
* add simple GR wavelength corrections
"""
