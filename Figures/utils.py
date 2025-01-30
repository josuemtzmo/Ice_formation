import os 
import glob
import xarray as xr
import numpy as np
from scipy import stats, signal
import datetime

def date_fmt(date):
    if len(date)>2:
        format = '%m-%d'
        outformat = '%d - %B'
    else:
        format = '%m'
        outformat = '%B'
    return datetime.datetime.strptime(str(date), format).strftime(outformat)

def okuboweissparm(u,v,grid,z=0,area_of_interest=dict(),grid_of_interest=None):
    area = area_of_interest.copy()
    grid_area = grid_of_interest.copy()
    
    du_dx = u.isel(area).diff('x').isel(y=grid_area['y'])/grid.e1u.isel(grid_area).squeeze()
    dv_dx = v.isel(area).diff('x').isel(y=grid_area['y'])/grid.e1v.isel(grid_area).squeeze()
    
    
    du_dy = u.isel(area).diff('y').isel(x=grid_area['x'])/grid.e2u.isel(grid_area).squeeze()
    dv_dy = v.isel(area).diff('y').isel(x=grid_area['x'])/grid.e2v.isel(grid_area).squeeze()
    
    sn = du_dx-dv_dy
    ss = dv_dx+du_dy
    ω = dv_dx-du_dy  
    owparm=sn**2+ss**2-ω**2
    return owparm.rename('Okubo_Weiss'),(sn,ss,ω)


def c_histogram(data,delta = 0.01,ignore_nan = True):

    data = data.where(data!=0)
    

    bins_int = np.arange(data.min(),data.max(),delta)
    n = len(bins_int)-1

    data_h=np.zeros((len(data.time_counter),n))
    bins_h=np.zeros((len(data.time_counter),n))


    counter=0
    for time in data.time_counter:
        data_snap = data.sel(time_counter = time)
        
        t_count = np.multiply(*data_snap.shape)
        
        ranget = (data_snap.min(),data_snap.max())
        
        if ignore_nan:
            data_snap = data_snap.values.ravel()
            data_snap_tmp = data_snap[~np.isnan(data_snap)]
            data_snap = data_snap_tmp
            
        freq,bins = np.histogram(data_snap,bins_int)
        data_h[counter,:] = freq/t_count
        bins_h[counter,:] = bins[:-1]

        counter+=1
        
    return data_h,bins_h


def xr_histogram(dataset,delta=None):
    delta_tmp = {var:0.01 for var in list(dataset.data_vars)}
    if not delta:
        delta = delta_tmp
    else:
        for key in delta_tmp.keys():
            var = list(delta.keys())[0]
            if var in key:
                delta_tmp[key]=delta[var]
            else:
                pass
        delta = delta_tmp
        
    datasets = []
    for var in list(dataset.data_vars):
        data_h,bins_h = c_histogram(dataset[var],delta[var])
        dataset2store = xr.Dataset( data_vars=dict( var=(["time","bin"], data_h),bbins=(["time","bin"], bins_h)), coords=dict(time=dataset.time_counter.values))
        datasets.append(dataset2store)
    
    return xr.merge(datasets)
        


def c_stats(data,ignore_nan = True):

    data_nan = data.where(data!=0)
    
    data_h=np.zeros((len(data_nan.time_counter)))
    bins_h=np.zeros((len(data_nan.time_counter)))

    data_stack = data_nan.stack(ravel=['x', 'y'])
            
    skewness = stats.skew(data_stack,axis=1,nan_policy='omit')
    kurtosis = stats.kurtosis(data_stack,axis=1,nan_policy='omit')
    standardev = data_stack.std('ravel')
    variance = data_stack.var('ravel')
    
    entropy = stats.entropy(data.stack(ravel=['x', 'y']),axis=1)
        
    return skewness,kurtosis,entropy,standardev,variance


#################################
############## FFT ##############
#################################

### Convert lon lat to x y coordinates
# P = pyproj.Proj(proj='utm', zone=31, ellps='WGS84', preserve_units=True)
def LonLat_To_XY(Lon,Lat):
    return gs.distance(Lon, Lat)

def  get_position_distance(dx_slice,dy_slice): 
    delta =  np.sqrt(dx_slice**2 + dy_slice**2)
    dist  =  np.cumsum(delta)
    return(dist,delta)


def field_template (var, dx, distance_cum):
    ## Interpolate on a new grid xn, with a cubic function
    xo = distance_cum
    field=var.assign_coords({var.dims[0]:xo.values})
    xn = np.arange(xo[0], xo[-1], step = dx)
    f_interp = field.interp({var.dims[0]:xn})
    return (xo, xn, field, f_interp)


def make_FFT1D(field,t,dt):
    ss =field
    #FFT=np.fft.rfft(ss,ss.shape[0])
    FFT=np.fft.fft(ss,ss.shape[0])
    #FFT=np.fft.fftshift(FFT)

    #freq1=np.fft.rfftfreq(len(t),dt)
    freq1=np.fft.fftfreq(len(t),dt)
    #freq1=np.fft.fftshift(freq1)

    ###ATTENTION, FFTFREQ RETOURNE UNE FREQUENCE DONC 1/DX
    om=2*np.pi*freq1
    #FFT=FFT*dt
    FFT=FFT/np.sqrt(ss.shape[0])
    PSD=FFT*np.conj(FFT)
    
    dk=dt
            ## Sampling frequency
    fs = 1/(dk)
            ## Number of points to overlap between segments (50%)
    nperseg = len(field)
    noverlap = nperseg/2
            # PSD computation
    f_welch, PSD_welch = signal.welch(ss, fs, nperseg=nperseg,window='hann', noverlap=noverlap, return_onesided=True, detrend='linear')
 

    return FFT,om, PSD, 2*np.pi*f_welch, PSD_welch

def load_data_oce(file):
    xfile = xr.open_dataset(file, decode_times=False)
    olon = xfile.nav_lon
    olat = xfile.nav_lat
    time = (xfile.time_counter)[0]
    depth = xfile.deptht
    temp = np.squeeze(xfile.votemper.isel(deptht=5))
    return(olon,olat,time,depth,temp)

def load_data_u(file):
    xfile = xr.open_dataset(file, decode_times=False)
    u = np.squeeze(xfile.vozocrtx.isel(depthu=5)).squeeze()
    return(u)

def load_data_v(file):
    xfile = xr.open_dataset(file, decode_times=False)
    v = np.squeeze(xfile.vomecrty.isel(depthv=5)).squeeze()
    return(v)

def load_grid(file):
    xfile = xr.open_dataset(file, decode_times=False)
    dx = xfile.e1t.squeeze()
    dy = xfile.e2t.squeeze()
    return (dx,dy)

def compute_PSD_noMeridionalAnom(data, dx=2000, idepth=1):
    FFT_data = np.zeros_like(data)
    PSD_welch_data = np.zeros_like(data)
    om_welch_data = np.zeros_like(data)
    for time in range(len(data.time_counter)):
        data_section = data.isel(time_counter=time) ### THE VARIABLE
#         data_mean = data_section.mean('x')

        var = data_section #- data_mean
        
        distance, dx_ = get_position_distance(np.ones(len(var))*dx,np.zeros(len(var)))

    #     xo, xn, field, field_tmp = field_template(var, dxx, distance)
        FFT,om, PSD, om_welch, PSD_welch = make_FFT1D(var,distance,dx)
        FFT_data[time,0:len(FFT)]=np.real(FFT)
        PSD_welch_data[time,0:len(PSD_welch)]=PSD_welch
        om_welch_data[time,0:len(om_welch)]=om_welch

    PSD_welch_data[PSD_welch_data==0]=np.nan
    
    return FFT_data, PSD_welch_data, om_welch_data

def compute_PSD(data, dx=2000, idepth=1):
    FFT_data = np.zeros_like(data)
    PSD_welch_data = np.zeros_like(data)
    om_welch_data = np.zeros_like(data)
    for time in range(len(data.time_counter)):
        data_section = data.isel(time_counter=time) ### THE VARIABLE
        data_mean = data_section.mean('x')

        var = data_section - data_mean
        
        distance, dx_ = get_position_distance(np.ones(len(var))*dx,np.zeros(len(var)))

    #     xo, xn, field, field_tmp = field_template(var, dxx, distance)
        FFT,om, PSD, om_welch, PSD_welch = make_FFT1D(var,distance,dx)
        FFT_data[time,0:len(FFT)]=np.real(FFT)
        PSD_welch_data[time,0:len(PSD_welch)]=PSD_welch
        om_welch_data[time,0:len(om_welch)]=om_welch

    PSD_welch_data[PSD_welch_data==0]=np.nan
    
    return FFT_data, PSD_welch_data, om_welch_data

def compute_PSD_dataset(dataset,indx=None):    
    dataset_dict = {}
    for var in list(dataset.data_vars):
        if indx!=None:
            FFT, PSD, om_welch = compute_PSD(dataset[var].isel(indx))
        else:
            FFT, PSD, om_welch = compute_PSD(dataset[var])
            
        dataset_dict[var+"_FFT"] = {'dims':('n','x'),'data': FFT} 
        dataset_dict[var+"_PSD"] = {'dims':('n','x'),'data':PSD,'attrs':{'long_name':dataset[var].long_name,'units':dataset[var].units}} 
        dataset_dict[var+"_om"] = {'dims':('n','x'),'data':om_welch}
    dataset = xr.Dataset.from_dict(dataset_dict)
    return dataset

#############################
### HEAT and SALT content ###
#############################


def heat_content(dataset, var_names, depth_var='deptht'):
    
    C_p = 3991.86
        
    list_vars = sorted([var for var in list(dataset.data_vars)[:]])
    var_uniques = list(np.unique([var.split('_')[0] for var in list_vars]))
    exp_uniques = list(np.unique([var.split('_')[-1] for var in list_vars]))
    exp_percent = np.array([ int(exp)*10**(2-np.round(np.log10(int(exp)))) if np.log10(int(exp)) < 1.0 else int(exp) for exp in exp_uniques ])
    exp_percent[np.isnan(exp_percent)]=0
    
    if len(var_names.keys()) <= 3:
        raise ValueError('The unique variables must include mix layer, density and temperature')
        
    heat_4_each_exp = []

    dims2expand = {key:item for key, item in dataset.dims.items() if 'y' in key or 'x' in key}
    depth_matrix = dataset[depth_var].expand_dims(dim=dims2expand).assign_coords({'x':dataset.x,'y':dataset.y})
    for expt in exp_uniques:
        mld_name = var_names['mld']+"_front_{0}".format(expt)
        toce_name = var_names['toce']+"_front_{0}".format(expt)
        rho_name = var_names['rhop']+"_front_{0}".format(expt)
        
        depth_mask_2_integrate = depth_matrix.where(depth_matrix <= dataset[mld_name],0).where(depth_matrix >= dataset[mld_name],1)
        
        heat_content = (C_p * ((273.15+dataset[toce_name])*dataset[rho_name]) * depth_mask_2_integrate).sum('deptht')
        

        heat_4_each_exp.append(heat_content.rename('MLD_heat_content_{0}'.format(expt)))

    return xr.merge(heat_4_each_exp)


#         mean_hc = heat_content.mean(('x','y')).compute()
#         var_hc = heat_content.var(('x','y')).compute()
    
#         heat_4_each_exp.append(var_hc.rename('MLD_varheat_{0}'.format(expt)))

def salt_content(dataset, var_names, depth_var='deptht'):
        
    list_vars = sorted([var for var in list(dataset.data_vars)[:]])
    var_uniques = list(np.unique([var.split('_')[0] for var in list_vars]))
    exp_uniques = list(np.unique([var.split('_')[-1] for var in list_vars]))
    exp_percent = np.array([ int(exp)*10**(2-np.round(np.log10(int(exp)))) if np.log10(int(exp)) < 1.0 else int(exp) for exp in exp_uniques ])
    exp_percent[np.isnan(exp_percent)]=0
    
    if len(var_names.keys()) <= 2:
        raise ValueError('The unique variables must include mix layer, density and temperature')
        
    salt_4_each_exp = []

    dims2expand = {key:item for key, item in dataset.dims.items() if 'y' in key or 'x' in key}
    depth_matrix = dataset[depth_var].expand_dims(dim=dims2expand).assign_coords({'x':dataset.x,'y':dataset.y})
    for expt in exp_uniques:
        mld_name = var_names['mld']+"_front_{0}".format(expt)
        soce_name = var_names['soce']+"_front_{0}".format(expt)

        depth_mask_2_integrate = depth_matrix.where(depth_matrix <= dataset[mld_name],0).where(depth_matrix >= dataset[mld_name],1)
        
        salt_content = (dataset[soce_name] * depth_mask_2_integrate).sum('deptht')
        

        salt_4_each_exp.append(salt_content.rename('MLD_salt_content_{0}'.format(expt)))

    return xr.merge(salt_4_each_exp)

#################################
############ IMPORT #############
#################################


class Import_Expt_Front():
    
    def __init__(self,folder_path):
        self.folder = folder_path
        self.exp_path = self.get_folders_in_path()
        
    def get_folders_in_path(self):
        exp_folders = os.listdir(self.folder)
        return [os.path.join(self.folder,exp_folder) for exp_folder in exp_folders if "expt" in exp_folder]

    def import_one_expt(self,expt,grid_file,freq='1d',**kwargs_xarray):
        if len(expt.split('/')) == 1:
            load_exp_path = [exp for exp in self.exp_path if expt == exp.split('/')[-1]][0]
        else: 
            load_exp_path = expt
        files_wildcard = load_exp_path+"/*{0}*{1}*.nc".format(freq,grid_file)
        list_of_files = glob.glob(files_wildcard)
        if len(list_of_files) == 1 and os.path.exists(list_of_files[0]):
            dataset = xr.open_dataset(list_of_files[0],**kwargs_xarray)
        elif len(list_of_files) > 1 and os.path.exists(list_of_files[0]):
            dataset = xr.open_mfdataset(files_wildcard,parallel=True,**kwargs_xarray)
        else:
            raise ValueError("File does not exist ({0}), check the expt and grid_file inputs.".format(list_of_files))
        return dataset
            
    def import_multipe_expt(self,grid_file,expt_filter=[],vars_of_interest=None,freq='1d',**kwargs_xarray):
        filtered_expt = [expt.split("/")[-1] for expt in self.exp_path]
        [filtered_expt.remove(expt) for expt in expt_filter]
        dataset_list = []
        for expt in filtered_expt:
            post_name = expt.split('expt_')[-1]
            if vars_of_interest:
                dataset = self.import_one_expt(expt,grid_file,freq,**kwargs_xarray)[vars_of_interest]
            else:
                dataset = self.import_one_expt(expt,grid_file,freq,**kwargs_xarray)
            rename_dict = {var:var+"_"+post_name  for var in list(dataset.data_vars)}
            dataset_list.append(dataset.rename(rename_dict))
            
        loaded_dataset = xr.merge(dataset_list).unify_chunks()

        return loaded_dataset
    
import datetime
from matplotlib import ticker
import shapely.plotting
from shapely.geometry import Polygon
import matplotlib.colors as mcolors

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def date_fmt_y_sm(date):
    """
    Function to format date
    """
    format = '%Y-%m-%d'
    return datetime.datetime.strptime(str(date), format).strftime('%d - %b')

def date_fmt_y(date):
    """
    Function to format date
    """
    format = '%Y-%m-%d'
    return datetime.datetime.strptime(str(date), format).strftime('%d - %B')

def date_fmt(date):
    """
    Function to format date
    """
    format = '%m-%d'
    return datetime.datetime.strptime(str(date), format).strftime('%d - %B')


def change_xticks(axis):
    """
    Function to change the xticks
    """
    xlabels=['0','100','300','500','700','900']
    ylabels=['0','50','250','400']

    axis.set_xticklabels(xlabels)
    axis.set_yticklabels(ylabels)

def plot_north_arrow( ax, xT=0, yT=0, scale=1, color='k' ):
    """
    Function to display arrow pointing to north.
    """
    def t_s(t,xT,yT,s):
        x,y = t
        return (xT+(x*s),yT+(y*s))
    a = [(0, 10), (0, 1), (2, 0)]
    b = [(0, 10), (0, 1), (-2, 0)]
    t_pos = (0.25,11.5)
    t_pos_x,t_pos_y = t_s(t_pos,xT,yT,scale)
    polygon1 = Polygon( [t_s(t,xT,yT,scale) for t in a] )
    polygon2 = Polygon( [t_s(t,xT,yT,scale) for t in b] )
    shapely.plotting.plot_polygon(polygon1, add_points=False, ax=ax, color=None, facecolor='None', edgecolor=color, linewidth=1)
    shapely.plotting.plot_polygon(polygon2, add_points=False, ax=ax, color=None, facecolor=color, edgecolor=color, linewidth=None)
    ax.text(x=t_pos_x,y=t_pos_y,s='N', fontsize=8,
            ha='center',
            va='center',weight='bold',color=color)
