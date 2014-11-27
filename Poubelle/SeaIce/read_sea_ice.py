from mpl_toolkits.basemap import Basemap, cm
# requires netcdf4-python (netcdf4-python.googlecode.com)
from netCDF4 import Dataset as NetCDFFile
import numpy as np
import matplotlib.pyplot as plt

def find(array, value):
    temp = []
    for i in array:
        idx = np.argmin(np.abs(i - value))
        temp.append((idx, i[idx]))

    temp2 = []
    for i in temp:
        temp2.append(i[1])

    ix = np.argmin(temp2)

    # Returns the index of array and its element
    # in the case: array[ix][temp[ix]]
    return ix, temp[ix]

f = open('bt_20130221_f17_v02_s.bin', 'rb')
seaIce = np.fromfile(f, dtype=np.uint16).reshape(332, 316)
f.close()
f = open('pss25lats_v3.dat', 'rb')
lats = np.fromfile(f, dtype='<i4').reshape(332, 316) / 100000.
f.close
f = open('pss25lons_v3.dat', 'rb')
lons = np.fromfile(f, dtype='<i4').reshape(332, 316) / 100000.
f.close()

north = -59.7183
south = -65.3099
west = -65.743
east = -48.55

ix_north = find(lats, north)
ix_south = find(lats, south)
ix_west = find(lons, west)
ix_east = find(lons, east)

inside = np.logical_and(np.logical_and(lons >= west,
                                       lons <= east),
                        np.logical_and(lats >= south,
                                       lats <= north))


def minmax(v):
    return np.min(v), np.max(v)

inds = np.where(inside)
imin, imax = minmax(inds[0])
jmin, jmax = minmax(inds[1])

ice_inside = seaIce[imin:imax+1, jmin:jmax+1]
lon = lons[imin:imax+1, jmin:jmax+1]
lat = lats[imin:imax+1, jmin:jmax+1]




# data from http://nsidc.org/data/docs/daac/nsidc0079_bootstrap_seaice.gd.html

#loncorners = -nc.variables['lon'][:]
#lon_0 = -nc.variables['true_lon'].getValue()
#lat_0 = nc.variables['true_lat'].getValue()
# create figure and axes instances
fig = plt.figure(figsize=(8,8))
ax = fig.add_axes([0.1,0.1,0.8,0.8])
# create polar stereographic Basemap instance.
#m = Basemap(projection='stere',lon_0=lon_0,lat_0=90.,lat_ts=lat_0,\
#            llcrnrlat=latcorners[0],urcrnrlat=latcorners[2],\
#            llcrnrlon=loncorners[0],urcrnrlon=loncorners[2],\
#            rsphere=6371200.,resolution='l',area_thresh=10000)

m = Basemap(llcrnrlon=-69.84,llcrnrlat=-70.35,\
	urcrnrlon=-40.11,urcrnrlat=-56.15,projection='merc',\
	resolution='l',area_thresh=10000)
# draw coastlines, state and country boundaries, edge of map.
m.drawcoastlines()
m.drawstates()
m.drawcountries()
# draw parallels.
parallels = np.arange(0.,90,10.)
m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
# draw meridians
meridians = np.arange(180.,360.,10.)
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
ny = data.shape[0]; nx = data.shape[1]
lons, lats = m.makegrid(nx, ny) # get lat/lons of ny by nx evenly space grid.
x, y = m(lons, lats) # compute map proj coordinates.
# draw filled contours.
clevs = [0,1,2.5,5,7.5,10,15,20,30,40,50,70,100,150,200,250,300,400,500,600,750]
cs = m.contourf(x,y,data,clevs,cmap=cm.s3pcpn)
# add colorbar.
cbar = m.colorbar(cs,location='bottom',pad="5%")
cbar.set_label('mm')
# add title
plt.title(prcpvar.long_name+' for period ending '+prcpvar.dateofdata)
plt.show()
