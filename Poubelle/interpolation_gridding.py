import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.mlab import griddata
from scipy.interpolate import Rbf
import numpy.ma as ma
from scipy.io import loadmat



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


X, Y = np.meshgrid(loni, lati)

tp0, ep0 = scaloa(X, Y, lons, lats, dat, corrlen=1, err=0.1**2)

fig, ax = make_map(bbox=bbox)
cs = ax.pcolormesh(X, Y, tp0)
cbar = fig.colorbar(cs, extend='both', shrink=0.85)
plt.show()


