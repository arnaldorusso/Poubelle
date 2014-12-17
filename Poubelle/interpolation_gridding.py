import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.mlab import griddata
from scipy.interpolate import Rbf
import numpy.ma as ma
from scipy.io import loadmat
from oceans.colormaps import cm



x = [-61.102, -58.8, -58.33, -58.2, -57.32, -59.54, -61.0, -57.8, -58.5]
y = [-63.08, -62.6, -63.065, -63.3, -62.2, -62.7, -63.4, -62.1, -63.02]

z = [33.3, 30.2, 31.5, 30, 33, 32, 28, 31.5, 33.6]

nx, ny = 50, 50
xi = np.linspace(-65, -50, nx)
yi = np.linspace(-59.8, -65.3, ny)
# zi = griddata(lons,lats,dat,xi,yi,interp='linear')

xi, yi = np.meshgrid(xi, yi)
xi, yi = xi.flatten(), yi.flatten()


def scipy_idw(x, y, z, xi, yi):
    interp = Rbf(x, y, z, function='linear')
    return interp(xi, yi)


zi = scipy_idw(x,y,z,xi,yi)
zi = zi.reshape((ny, nx))

plt.imshow(zi) #, extent=(x[-1], x[0], y[-1], y[0]))
#plt.contourf(xi,yi,zi,cmap=plt.cm.rainbow)
#plt.scatter(x,y,c=z)
plt.colorbar()
plt.show()

### Basemap
lon = [-61.102, -58.8, -58.33, -58.2, -57.32, -59.54, -61.0, -57.8, -58.5]
lat = [-63.08, -62.6, -63.065, -63.3, -62.2, -62.7, -63.4, -62.1, -63.02]
data = [33.3, 30.2, 31.5, 30, 33, 32, 28, 31.5, 33.6]

loni = np.linspace(-65, -50, 200)
lati = np.linspace(-59.8, -65.3, 200)
datai = griddata(lon, lat, data, loni, lati, interp='linear')

fig, ax = plt.subplots()
m = Basemap(projection='mill',
            llcrnrlon=-63,
            llcrnrlat=-65.5,
            urcrnrlon=-53,
            urcrnrlat=-59.8,
            resolution='h')

m.drawcoastlines()
m.drawmapboundary()
m.fillcontinents()
m.drawparallels(np.arange(-65, -59, 1), labels=[1,0,0,0])
m.drawmeridians([-60, -55], labels=[0, 0, 0, 1])

# A principal diferença está aqui.  Estou usando o `contourf` do object do Basemap e não do matplotlib.
loni, lati = np.meshgrid(loni, lati)
cs = m.contourf(loni, lati, datai, cmap=plt.cm.rainbow, zorder=1, latlon=True)
#cm = m.pcolormesh(loni, lati, datai, cmap=plt.cm.rainbow, zorder=1, latlon=True)
sc = m.scatter(lon, lat, marker='o', c='k', edgecolors='none', zorder=2, latlon=True)

ax.set_title('griddata test')
cbar = fig.colorbar(cs)

plt.show()


#### MORE Tests
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

merc = loadmat('/home/yepan/Downloads/mercator_temperature.mat', squeeze_me=True)

LAND = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                    edgecolor='face',
                                    facecolor=cfeature.COLORS['land'])

def make_map(bbox, projection=ccrs.PlateCarree()): #Mercator, LambertCylindrical
    fig, ax = plt.subplots(figsize=(8, 6),
                           subplot_kw=dict(projection=projection))
    ax.set_extent(bbox)
    ax.add_feature(LAND, facecolor='0.75')
    ax.coastlines(resolution='50m')
    gl = ax.gridlines(draw_labels=True)
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    return fig, ax



def scaloa(xc, yc, x, y, data, corrlen=1.5, err=0.09**2):
    shape = xc.shape
    xc, yc, x, y, data = map(np.ravel, (xc, yc, x, y, data))
    n = len(x)
    x, y = np.reshape(x, (1, n)), np.reshape(y, (1, n))
    # Squared distance matrix between the observations.
    d2 = ((np.tile(x, (n, 1)).T - np.tile(x, (n, 1))) ** 2 +
          (np.tile(y, (n, 1)).T - np.tile(y, (n, 1))) ** 2)
    nv = len(xc)
    # Squared distance between the observations and the grid points.
    dc2 = ((np.tile(xc, (n, 1)).T - np.tile(x, (nv, 1))) ** 2 +
           (np.tile(yc, (n, 1)).T - np.tile(y, (nv, 1))) ** 2)
    # Correlation matrix between stations (A) and cross correlation (stations
    # and grid points (C))
    A = (1 - err) * np.exp(-d2 / corrlen ** 2)
    C = (1 - err) * np.exp(-dc2 / corrlen ** 2)
    # Add the diagonal matrix associated with the sampling error.  We use the
    # diagonal because the error is assumed to be random.  This means it just
    # correlates with itself at the same place.
    A = A + err * np.eye(len(A))
    # Weights that minimize the variance (OI).
    data = np.reshape(data, (n, 1))
    tp = np.dot(C, np.linalg.solve(A, data))
    # Normalized mean error.  Taking the squared root you can get the
    # interpolation error in percentage.
    ep = 1 - np.sum(C.T * np.linalg.solve(A, C.T), axis=0) / (1 - err)
    tp = tp.reshape((shape))
    ep = ep.reshape((shape))
    return tp, ep


lons = np.array([-61.102, -58.8, -58.33, -58.2, -57.32, -59.54, -61.0, -57.8, -58.5])
lats = np.array([-63.08, -62.6, -63.065, -63.3, -62.2, -62.7, -63.4, -62.1, -63.02])
dat = np.array([33.3, 30.2, 31.5, 30, 33, 32, 28, 31.5, 33.6])
#X, Y = np.meshgrid(lons, lats)

bbox = [-63, -53, -65.5, -59.8]
loni = np.linspace(-63, -53, 300)
lati = np.linspace(-65.5, -59.8, 300)
datai = griddata(lons, lats, dat, loni, lati, interp='linear')

fig, ax = make_map(bbox=bbox)
cs = ax.pcolormesh(loni, lati, datai)
cbar = fig.colorbar(cs, extend='both', shrink=0.85)
plt.show()


## Performing Interpolation

def make_ax(ax, tp):
    ax.set_extent(bbox)
    ax.add_feature(LAND, facecolor='0.75')
    ax.coastlines(resolution='50m')
    gl = ax.gridlines(draw_labels=True)
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    cs = ax.pcolormesh(X, Y, tp)
    return ax, cs


# Initial griding
X, Y = np.meshgrid(loni, lati)


tp0, ep0 = scaloa(X, Y, lons, lats, dat, corrlen=1, err=0.1**2)

fig, ax = make_map(bbox=bbox)
cs = ax.pcolormesh(X, Y, tp0)
cbar = fig.colorbar(cs, extend='both', shrink=0.85)
plt.show()


#### More about gridding
def ll2km(lon, lat, bbox):
    """xkm, ykm will be the coordinates of the data (converted from lon/lat)."""
    rearth = 6370800  # Earth radius [m].
    deg2rad = np.pi/180
    ykm = rearth * (lat - bbox[2]) * deg2rad / 1000
    xkm = rearth * ((deg2rad * (lon - bbox[0])) *
                    np.cos(deg2rad * 0.5 * (lat + bbox[2])) / 1000)

    return xkm, ykm


def func(a, x, fx, method='markov'):
    """Compute the mean squared misfit between an analytical function
    (e.g. Gaussian or Markov function) and a set of data FX observed at
    independent coordinates X.
    
    http://marine.rutgers.edu/dmcs/ms615/jw_matlab/oi/myfunction.m

    Parameters
    ----------
    a : float
        Parameters of analytically function
    x : array
        Locations of data
    fx : array
         Observed function values (data).

    method : string
            Specifies the shape of the function to be fitted:
            Must be one of 'gauss', 'markov' (default) or 'letra' (LeTraon).

    In all cases, two parameters are fit a[0] is the y-intercept at x=0
    a[1] is the characteristic scale of the fitted function.
    """

    # Gaussian function f = a0 * exp(-0.5 * (r/a)**2).
    if method == 'gauss':
        r = x / a[1]
        fit = a[0] * np.exp(-0.5 * r**2)
    # Markov function f = a0 * (1 + r/a) * exp(-r/a).
    elif method == 'markov':
        r = np.abs(x) / a[1]
        fit = a[0] * (1+r) * np.exp(-r)
        # Le Traon function f = a0 * exp(-r/a) * (1+r/a+(r**2) / 6-(r**3) / 6.
    elif 'letra':
        r = np.abs(x) / a[1]
        rsq = r**2
        fit = a[0] * np.exp(-r) * (1 + r + rsq / 6 - (r * rsq) / 6)
    else:
        raise ValueError("Unrecognized method {!r}.  Must be one of 'gauss',"
                         " 'markov' or 'letra'.".format(method))

    return np.mean((fit - fx)**2)


def optinter(R, lamb, x, y, data, xdata, ydata, cov_func='markov'):
    """The code uses optimal interpolation to map irregular spaced observations
    onto a regular grid.
    
    http://marine.rutgers.edu/dmcs/ms615/jw_matlab/oi/oi_mercator.m

    Parameters
    ----------
    R : float
        Square root of the de-correlation length scale in units of deg**2.
    lambda : float
                 error squared to signal squared or E.
    X, Y : array
           Grid of the locations for theta.
    data : array
           Observations.
    xdata, ydata : array
           Observed locations.

    Returns
    -------
    theta : array
            Optimal interpolated data.
    err : array
          Estimated optimum error.
    res : array
          Residue fit.
    """
    X, Y = ll2km(*np.meshgrid(x, y), bbox=bbox)
    
    # Ars.
    xkm, ykm = ll2km(xdata, ydata, bbox)
    xr, yr = np.meshgrid(xkm, ykm)

    # Compute the distance of each data point:
    rdist = np.sqrt((xr - xr.T)**2 + (yr - yr.T)**2)

    # Covariance function.
    if cov_func == 'gauss':
        cdd0 = np.exp(-rdist**2 / R**2)
    elif cov_func == 'markov':
        cdd0 = (1 + rdist/R) * np.exp(-rdist/R)
    else:
        raise ValueError("Unrecognized covariance function {!r}."
                         "Must be one of 'gauss' or 'markov'.".format(cov_func))

    # Final Data covariance Matrix between data points.
    cdd = cdd0 + lamb * np.eye(*cdd0.shape)

    # Cxr.
    Xd, Xg = np.meshgrid(xkm, X.ravel())
    Yd, Yg = np.meshgrid(ykm, Y.ravel())

    # Distance between observation r and grid g.
    rmd = np.sqrt((Xg - Xd)**2 + (Yg - Yd)**2)

    # Again plug into covariance function.
    if cov_func == 'gauss':
        cmd = np.exp(-rmd**2 / R**2)
    elif cov_func == 'markov':
        cmd = (1 + rmd /R) * np.exp(-rmd/R)
    else:
        raise ValueError("Unrecognized covariance function {!r}."
                         "Must be one of 'gauss' or 'markov'.".format(cov_func))

    demeaned =  data - data.mean()
    res = data.mean() + np.dot(cdd0, np.linalg.solve(cdd, demeaned))
    res = data.ravel() - res.ravel()

    # Normalized by the error variance.
    # err = np.diag(1 - np.dot(np.dot(cmd, np.linalg.inv(cdd)), cmd.T))
    # err = np.reshape(err, X.shape) * 100  # Error in percentages.

    theta = data.mean() + np.dot(cmd, np.linalg.solve(cdd, demeaned))
    theta = np.reshape(theta, X.shape)

    return dict(residual=res, 
#		error=err,
		covariance=cdd0,
		final_covariance=cdd,
		theta=theta)


for r in rs[1:]:
    idx = np.logical_and(Rd.ravel() > r - 0.5 * dr, Rd.ravel() <= r + 0.5 * dr)
    cf.append(C.ravel()[idx].mean())


R = 70  # [km] Overestimation.
lamb = 0.11

ret = optinter(R, lamb, loni, lati, dat, lons, lats, cov_func='markov')

fig, ax = make_map(bbox=bbox)
cs = ax.pcolormesh(X, Y, ret['theta'], cmap=cm.avhrr)
ax.set_title('Too smooth!')
cbar = fig.colorbar(cs, extend='both', shrink=0.85)
cdot = ax.scatter(lon, lat, marker='o', c='k', edgecolors='none', zorder=2)
plt.show()

### Inserting error to compute Markov Chain
tdat = dat.copy()
e = 0.75
tdat += e * np.random.randn(dat.size)

demaned = tdat - tdat.mean()
C = np.dot(demaned[:, None], demaned[:, None].T)  # Cov.

# Distance to check for the scale.
dr = 30  # Distance step.
rs = np.r_[0, np.arange(0.5*dr, 100, dr)]

cf = [np.diag(C).mean()]
XKM, YKM = ll2km(*np.meshgrid(lons, lats), bbox=bbox)
Rd = np.sqrt((XKM-XKM.T)**2 + (YKM-YKM.T)**2)
for r in rs[1:]:
    idx = np.logical_and(Rd.ravel() > r - 0.5 * dr,
                         Rd.ravel() <= r + 0.5 * dr)
    cf.append(C.ravel()[idx].mean())

cf = np.array(cf)  # Observed Covariance Function.
