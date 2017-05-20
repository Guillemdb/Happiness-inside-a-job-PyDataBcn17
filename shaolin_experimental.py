import pandas as pd
import numpy as np
from shaolin.core.dashboard import Dashboard, ToggleMenu
from shaolin.dashboards.slicers import PanelSlicer, Panel4DSlicer, DataFrameSlicer
from shaolin.dashboards.data_transforms import DataFrameScaler
from shaolin.dashboards.colormap import ColormapPicker
from io import BytesIO
import base64
import warnings
import datetime

class AxisPlotLight(Dashboard):
    def __init__(self,
                 data,
                 mask,
                 strf='%a\n%d\n%h\n%Y',
                 font_size=20,
                 pct_space=0.1,
                 width="20em",
                 height="3em",
                 cmap=None,
                 orient='horizontal',
                 **kwargs
                ):
        if cmap is None:
            colors = diverging_palette(h_neg=38, h_pos=167, s=99, l=66, sep=39, n=256, center='light')
            cmap = LinearSegmentedColormap.from_list("interactive", colors)
            cmap._init()
            cmap._set_extremes()
            self.cmap = cmap
        else:
            self.cmap = cmap
            
        self._width = width
        self._height = height
        self.data = data
        self.mask = mask
        if self.is_time_array(data):
            start = 'HTML$N=start&val='+pd.to_datetime(self.data[0]).strftime(strf)                   
            end ='HTML$N=end&val='+pd.to_datetime(self.data[-1]).strftime(strf)               
            #self.start = TimeDisplay(datetime=pd.to_datetime(self.data[0]), strf=self.strf.value)
            #self.end = TimeDisplay(datetime=pd.to_datetime(self.data[-1]), strf=self.strf.value)
        else:
            start = 'HTML$N=start&val='+str(self.data[0])
            end = 'HTML$N=end&val='+str(self.data[-1])
        self.strf = strf
        self.font_size = font_size
        self.pct_space = pct_space
        self.orient = orient
            
        plot = 'HTML$N=plot'
        dash = ['c$N=AxisPlot',
                 [
                    ['r$N=plot_box',[start, plot, end]]
                 ]
               ]
        Dashboard.__init__(self, dash, **kwargs)        
        self.plot.value = self.plot_axis(data=self.data,
                                   idx=self.mask,
                                   cmap=self.cmap,
                                   strf=self.strf,
                                   fontsize=self.font_size,
                                   pct_space=self.pct_space
                                  )
        self.observe(self._trigger_update)
        
        self._trigger_update()
        self.update()
    
    def _trigger_update(self, _=None ):
        try:
            self.update()
        except ValueError:#when there is not a valid strf while typing 
            pass
    def update(self, data=None, mask=None):
        if data is None:
            data = self.data
        if mask is None:
            mask = self.mask
        self.plot.value = self.plot_axis(data,
                                   idx=mask,
                                   cmap=self.cmap,
                                   strf=self.strf,
                                   fontsize=self.font_size,
                                   pct_space=self.pct_space
                                  )
        self.plot.widget.layout.width =self._width
        self.plot.target.layout.width = "100%"
        self.plot.target.layout.max_width = "15em"
        self.plot.widget.layout.height = self._height
        self.plot.target.layout.height = "100%"
        self.plot.target.layout.max_height = "5em"
        self.data = data
        self.mask = mask
        if self.is_time_array(data):
            self.start.target.description = pd.to_datetime(self.data[0]).strftime(self.strf)
            self.end.target.description = pd.to_datetime(self.data[-1]).strftime(self.strf)
        else:
            self.start.target.description = str(self.data[0])
            self.end.target.description = str(self.data[-1])
        
        
    @staticmethod
    def is_time(val):
        int_or_float = (int, np.int, np.int16, np.int0, np.int8,
                        np.int32, np.int64, np.float, np.float128,
                        np.float16, np.float32, np.float64, float)
        time_type = (datetime.date, datetime.datetime, np.datetime64)
        
        if isinstance(val,time_type):
            return True
        elif isinstance(val,int_or_float):
            return False
        try:
            _ = pd.to_datetime(val)
            _ = val.strftime('%Y')
            return True
        except ValueError:
            return False 
        except AttributeError:
            return False
        
    @staticmethod
    def is_time_array(val):
        return np.all([AxisPlotLight.is_time(x) for x in val])
    
    def fig_to_html(self, fig, img_class=''):
        """This is a hack that converts a matplotlib like fig to an HTML image div"""
        warnings.filterwarnings('ignore')
        imgdata = BytesIO()
        fig.savefig(imgdata, format='png',bbox_inches='tight',dpi=160, transparent=True)
        imgdata.seek(0)  # rewind the data
        svg_dta = imgdata.getvalue()
        svg_image = Image(svg_dta)#
        svg_b64 = base64.b64encode(svg_image.data).decode()
        css = '''<style>
                        .cmap-axis {width:'''+self._width+''' !important;
                               height:'''+self._height+'''em !important;
                               margin:15px;
                               align:left;}

                </style>'''
        height = self.plot.target.layout.height
        width = self.plot.target.layout.width
        return '<img class="cmap '+img_class+'" height="'+str(height)+'" width="'\
                +str(width)+'" src="data:image/png;base64,'+svg_b64+'" />'
        
        #return '<img class="cmap-axis" '+img_class+'" src="data:image/png;base64,'+svg_b64+'" />'#+css
    
    
        
    def plot_axis(self,data,idx,cmap,strf='%a\n%d\n%h\n%Y',fontsize=20, pct_space=0.1):
        #display original array as index
        xticks_ix = int(math.ceil(len(data)/100)*1/pct_space)
        xticks_vals = np.arange(len(data))[::xticks_ix]
        xticks_labels = data[::xticks_ix]
        with axes_style("white"):
                f, ax = plt.subplots(figsize=(10, 1))
        if True:#idx.ndim <= 1:
            idx_plot = np.tile(idx, (1, 1))
        else:
            idx_plot = idx
            
        ax.pcolormesh(idx_plot,cmap=cmap)
        if self.is_time_array(data):
            ticks = pd.DatetimeIndex(xticks_labels).to_pydatetime()
            _ = plt.xticks(xticks_vals, [x.strftime(strf) for x in ticks], fontsize = fontsize,
                           rotation=self.orient)
        else:
            _ = plt.xticks(xticks_vals, xticks_labels, fontsize = fontsize,
                           rotation=self.orient)
        ax.axis([0,len(data), 0, 1])
        ax.axes.get_yaxis().set_visible(False)
        #f.tight_layout()

        html =  self.fig_to_html(f,img_class='axisplot')
        plt.close(f)
        return html

class ArraySlicer(Dashboard):
    """Array slicer widget. Performs arbitary slicing in a array-like object"""
    def __init__(self,
                 data,
                 start=0,
                 end=-1,
                 step=1,
                 display='sliders',
                 slice_mode='single',
                 description='Slicer',
                 strf='%d\n%h\n%Y\n%T',
                 **kwargs):
        self.strf = strf
        self.halt_update = False
        if AxisPlotLight.is_time_array(data):
            st = str(self.strf)
            st = st.replace('\n',' ')
            time_data = [pd.to_datetime(x).strftime(st) for x in data]
            dd_sel = ['@sel$N=dd_selector&o='+str(list(time_data))]
            sel_sli = ['@selslider$N=sel_sli&o='+str(list(time_data))]
            self._data = time_data
            self._ast_vals = False
        
        elif isinstance(data[0],str):
            dd_sel = ['@sel$N=dd_selector&o='+str(list(data))]
            sel_sli = ['@selslider$N=sel_sli&o='+str(list(data))]
            self._data = data
            self._ast_vals = False
        else:
            self._data = [str(x) for x in data]
            dd_sel = ['@sel$N=dd_selector&o=["'+str(list(data)[0])+'"]']
            sel_sli = ['@selslider$N=sel_sli&o=["'+str(list(data)[0])+'"]']
            self._ast_vals = True
        if end==-1:
            end = len(data)
        
        self.idx = np.ones(len(data),dtype=bool)
        plot = AxisPlotLight(data,self.idx, name='plot',strf=strf)
        dash = ['c$N=array_slicer',
                [['###'+description+'$N=title_slicer'],
                 ['r$N=plot_row',[plot]],
                 dd_sel,
                 sel_sli,
                 ['r$N=controls_row',[['r$N=columns_mode_row',['@togs$N=slice_mode&o=["single", "slice"]&val='+str(slice_mode),
                                                               ]],


                                      ['c$N=sliders_col',['@('+str(-len(data)+1)+','+str(end)+',1,'+str(start)+')$N=start_slider&d=Start',
                                                          '@('+str(-len(data)+1)+','+str(end)+',1,'+str(end)+')$N=end_slider&d=End',
                                                          '@(1,'+str(len(data)-1)+',1,'+str(step)+')$N=step_slider&d=Step'
                                                         ]

                                      ],
                                     ]
                 ]
                ]
               ]    
        Dashboard.__init__(self, dash, mode='interactive', **kwargs)
        
        if self._ast_vals:
            self.dd_selector.target.options = [str(x) for x in data]
            self.sel_sli.target.options = [str(x) for x in data]
        self.sel_sli.target.continuous_update=False
        self.start_slider.target.continuous_update=False
        self.dd_selector.target.layout.width = "12em"
        self.dd_selector.observe(self._link_dropdown)
        self.sel_sli.observe(self._link_sel_sli)
        self.start_slider.observe(self._link_start_sli)
        self.observe(self.update)
        self.update()
        self.data = self._data.copy()
     
    @property
    def selector_vals(self):
        sel_vals = {'column':self.dd_selector.value,
                    'start':self.start_slider.value,
                    'end':self.end_slider.value,
                    'step':self.step_slider.value
                   }
        return dict(sel_vals)
    
    def update_selectors(self,**kwargs):
        self.halt_update = True
        if 'start' in kwargs:
            self.start_slider.value = kwargs['start']
            self._link_start_sli()
        if 'column' in kwargs:
            if str(kwargs['column']) in self.dd_selector.target.options:
                self.dd_selector.target.value = str(kwargs['column'])
                self._link_dropdown()
        if 'end' in kwargs:
            self.end_slider.value = kwargs['end']
        if 'step' in kwargs:
            self.step_slider.value = kwargs['step']
        self.halt_update = False
        self.update()
        
    def _link_dropdown(self, _=None):
        try:
            self.sel_sli.value = self.dd_selector.value
            self.start_slider.value = list(self._data).index(self.dd_selector.value)
        except:
            pass
    
    def _link_sel_sli(self, _=None):
        try:
            self.dd_selector.value = self.sel_sli.value  
            self.start_slider.value = list(self._data).index(self.dd_selector.value)
        except:
            pass
        
    def _link_start_sli(self, _=None):
        try:
            if self.start_slider.value >= len(self._data):
                self.dd_selector.value = list(self._data)[-1]
                self.sel_sli.value = list(self._data)[-1]
            else:
                self.dd_selector.value = list(self._data)[self.start_slider.value]
                self.sel_sli.value = list(self._data)[self.start_slider.value]
        except:
            pass
    
    @property
    def data(self):
        return self._data
    
    
    
    @data.setter
    def data(self, val):
        new_end = len(val)
        old_end = len(self._data)
        self._data = val
        self.halt_update = True
        if new_end != old_end:

            if AxisPlotLight.is_time_array(val):

                st = str(self.strf) 
                st = st.replace('\n',' ')
                time_data = [pd.to_datetime(x).strftime(st) for x in val]
                self._data = time_data
                if old_end < new_end:
                    self.dd_selector.target.options = list(time_data)
                    self.dd_selector.target.value = list(time_data)[0]
                else:
                    
                    self.dd_selector.target.value = list(time_data)[0]
                    self.dd_selector.target.options = list(time_data)
                    
            elif self._ast_vals:
                self._data = [str(x) for x in val]
                if old_end < new_end:
                    self.dd_selector.target.options = [str(x) for x in self._data]
                    self.dd_selector.target.value = str(list(self._data)[0])
                else:
                    self.dd_selector.target.value = str(list(self._data)[0])
                    self.dd_selector.target.options = [str(x) for x in self._data]
            else:
               
                if old_end < new_end:
                    self.dd_selector.target.options = list(self._data)
                    self.dd_selector.target.value = list(self._data)[0]
                else:
                    self.dd_selector.target.value = list(self._data)[0]
                    self.dd_selector.target.options = list(self._data)
            if old_end < new_end:
                self.start_slider.target.max = new_end
                self.dd_selector.target.options = list(self._data)
                self.dd_selector.target.value = list(self._data)[0]
                #self.start_slider.target.value = new_end
                self.start_slider.target.min = -new_end +1
                self.end_slider.target.max = new_end
                self.end_slider.target.value = new_end
                self.end_slider.target.min = -new_end +1   
            else:
                self.start_slider.target.max = new_end
                self.start_slider.target.min = -new_end +1
                self.dd_selector.target.value = list(self._data)[0]
                self.dd_selector.target.options = list(self._data)
                #self.dd_selector.target.value = list(self._data)[0]
                #self.start_slider.target.value = new_end
                
                self.end_slider.target.value = new_end
                self.end_slider.target.max = new_end
                self.end_slider.target.min = -new_end +1            
            self.step_slider.target.max = new_end-1
        #self.sel_sli.options = list(self._data)
        self._link_dropdown()
        self.halt_update = False
        self.update()
        
    def filter_index(self):
        new_idx = np.zeros(len(self._data), dtype=bool)
        if self.slice_mode.value == 'single':
            if len(self.data) <=5000:
                new_idx[list(self._data).index(self.dd_selector.value)] = True
            else:
                if self.start_slider.value < len(self._data):
                    new_idx[self.start_slider.value] = True
                else:
                    new_idx[-1] = True
        else:
            if self.end_slider.value <= len(self._data)-1:
                 new_idx[self.start_slider.value:\
                        self.end_slider.value:\
                        self.step_slider.value] = True
            else:
                new_idx[self.start_slider.value::self.step_slider.value] = True
                #new_idx[-1] = True
        self.idx = new_idx        
    
    
    def update_mode(self):
        self.halt_update = True
        if self.slice_mode.value == 'single':
            if len(self.data) <=5000:
                self.plot_row.visible = False
                self.dd_selector.visible = True
                self.sel_sli.visible = True
                self.start_slider.visible = False
            else:
                self.start_slider.visible = True
                self.plot_row.visible = True
                self.dd_selector.visible = False
                self.sel_sli.visible = True
            self.end_slider.visible = False
            self.step_slider.visible = False
        else:
            self.plot_row.visible = True
            self.dd_selector.visible = False
            self.sel_sli.visible = True
            self.end_slider.visible = True
            self.step_slider.visible = True
            self.start_slider.visible = True
        self.halt_update = False

    def update_widgets(self):
        L = len(self._data)
        for w in ['start_slider', 'end_slider', 'step_slider']:
            if w != 'step_slider':
                getattr(self, w).target.min = -L
            getattr(self, w).target.max = L
        self.update_mode()
        self.plot.update(self.data, self.idx)


    def update(self,_=None, data=None):
        if not self.halt_update:
            if not data is None:
                self._data = data
                self.idx = np.ones(len(data),dtype=bool)
            self.filter_index()
            self.update_widgets()


class DataFrameSlicer(Dashboard):
    
    def __init__(self, df,description='DataFrame', **kwargs):
        self.df = df
        self.output = None
        dash =['r$N=df_slicer',
               [
                '##'+description+'$N=description_text',
                ArraySlicer(df.index.values, name='index_slicer', description='index'),
                ArraySlicer(df.columns.values, name='columns_slicer', description='columns')
               ]
              ]
        Dashboard.__init__(self, dash, mode='interactive', **kwargs)
        self.observe(self.update)
        self.update()
    
    @property
    def selector_vals(self):
        vals = {}
        ix_vals = self.index_slicer.selector_vals
        for key,value in ix_vals.items():
            vals[key+'_ix'] = value
        cols_vals = self.columns_slicer.selector_vals
        for key,value in cols_vals.items():
            vals[key+'_col'] = value
        return vals
    
    def update_selectors(self,**kwargs):
        ix_kwrgs = {}
        col_kwrgs = {}
        if 'start_ix' in kwargs:
            ix_kwrgs['start'] = kwargs['start_ix']
        if 'column_ix' in kwargs:
            ix_kwrgs['column'] = kwargs['column_ix']
        if 'end_ix' in kwargs:
            ix_kwrgs['end'] = kwargs['end_ix']
        if 'step_ix' in kwargs:
            ix_kwrgs['step'] = kwargs['step_ix']
        self.index_slicer.update_selectors(**ix_kwrgs)
        if 'start_col' in kwargs:
            col_kwrgs['start'] = kwargs['start_col']
        if 'column_col' in kwargs:
            col_kwrgs['column'] = kwargs['column_col']
        if 'end_col' in kwargs:
            col_kwrgs['end'] = kwargs['end_col']
        if 'step_col' in kwargs:
            col_kwrgs['step'] = kwargs['step_col']
        self.columns_slicer.update_selectors(**col_kwrgs)
        self.update()
    @property
    def description(self):
        return self.df.columns[self.columns_slicer.idx].values[0]
    
    @property
    def data(self):
        return self.df
    @data.setter
    def data(self, val):
        self.df = val
        self.index_slicer.data = val.index.values
        self.columns_slicer.data = val.columns.values
        self.update()
        
    def _init_layout(self):
        self.index_slicer.plot.img_height.value = 3.
        self.index_slicer.plot.img_width.value = 20.
        self.columns_slicer.plot.img_height.value = 3.
        self.columns_slicer.plot.img_width.value = 20.
    
    def update(self, _=None):
        self.output = self.df.ix[self.index_slicer.idx,self.columns_slicer.idx]

from bokeh.sampledata.iris import flowers
from seaborn.palettes import diverging_palette
from matplotlib.colors import LinearSegmentedColormap

from IPython.display import Image
from seaborn.rcmod import axes_style 
import matplotlib.pyplot as plt

from shaolin.core.dashboard import Dashboard
from shaolin.dashboards.colormap import MasterPalette
import math


import numpy as np
import pandas as pd
from shaolin.core.dashboard import Dashboard
class DataFrameScaler(Dashboard):
    
    def __init__(self,
                 data,
                 funcs=None,
                 min=0,
                 max=100,
                 step=None,
                 low=None,
                 high=None,
                 **kwargs):
        self.halt_update = False
        if funcs is None:
            self.funcs = {'raw':lambda x: x,
                          'zscore': lambda x: (x-np.mean(x))/np.std(x),
                          'log': np.log,
                          'rank':lambda x: pd.DataFrame(x).rank().values.flatten(),
                          'inv': lambda x: -x
                         }
        else:
            self.funcs  = funcs
        self._df = data.apply(self.categorical_to_num)
        if min is None:
            min = self._df.min().values[0]
        if max is None:
            max = self._df.max().values[0]
        if step is None:
            step = (max-min)/100.
        if low is None:
            low = min
        if high is None:
            high = max
        self.halt_update = False
        self.output = None
        
        dash = ['c$N=df_scaler',
                ['@('+str(min)+', '+str(max)+', '+str(step)+', ('+str(low)+', '+str(high)+'))$N=scale_slider&d=Scale',
                 ['r$N=main_row',['@dd$d=Apply&N=dd_sel&val=raw&o='+str(list(self.funcs.keys())),'@True$N=scale_chk&d=Scale']]
                ]
               ]
        Dashboard.__init__(self, dash, mode='interactive', **kwargs)
        self.dd_sel.target.layout.width = "100%"
        self.scale_chk.widget.layout.padding = "0.25em"
        self.observe(self.update)
        self.update()
    
    @property
    def selector_vals(self):
        #low,high = self.scale_slider.value
        vals = {'min':self.scale_slider.target.min,
                'max':self.scale_slider.target.max,
                'value':self.scale_slider.value,
                'apply':self.dd_sel.value,
                'scale':self.scale_chk.value
               }
        return dict(vals)
    
    def update_selectors(self,**kwargs):
        self.halt_update = True
        if 'low' in kwargs:
            low = kwargs['low']
        elif 'value' in kwargs:
            low = kwargs['value'][0]
        else:
            low = self.scale_slider.value[0]
        if 'high' in kwargs:
            high = kwargs['high']
        elif 'value' in kwargs:
            high = kwargs['value'][1]
        else:
            high = self.scale_slider.value[1]
            
        if 'min' in kwargs: 
            if kwargs['min'] <= self.scale_slider.value[0]:
                self.scale_slider.target.min = kwargs['min']
            else:
                self.scale_slider.target.max = max(kwargs['min'],self.scale_slider.value[1])
                self.scale_slider.value = (kwargs['min'],max(kwargs['min'],self.scale_slider.value[1]))
                self.scale_slider.target.min = kwargs['min']
        if 'max' in kwargs:
            if kwargs['max'] >= self.scale_slider.value[1]:
                self.scale_slider.target.max = kwargs['max']
            else:
                self.scale_slider.value = (kwargs['min'],kwargs['max'])
                self.scale_slider.target.max = kwargs['max']
            
        
        self.scale_slider.value = (low,high)
        if 'apply' in kwargs:
            self.dd_sel.value = kwargs['apply']
        if 'scale' in kwargs:
            self.scale_chk.value = kwargs['scale']
        self.halt_update = False
        self.update()
    @property
    def data(self):
        return self._df

    @data.setter
    def data(self, val):
        self._df = val.apply(self.categorical_to_num)
        self.update()
    
    def scale_func(self, data):
        Ma = np.max(data)
        mi = np.min(data)
        score = ((data-mi)/(Ma-mi))#to 0-1 interval
        if mi == Ma:
               return np.ones(len(score)) *0.5
        scale_h = self.scale_slider.value[1]
        scale_l = self.scale_slider.value[0]
        return score*(scale_h-scale_l)+scale_l 
    
    def update(self, _=None):
        if not self.halt_update:
            self.output = self.data.apply(self.funcs[self.dd_sel.value])
            if self.scale_chk.value:
                self.scale_slider.visible = True
                self.output = self.output.apply(self.scale_func)
            else:
                self.scale_slider.visible = False
    @staticmethod
    def categorical_to_num(data):
        """Converts categorical data into an array of ints"""
        if isinstance(data.values[0], (str,bool)):
            if isinstance(data.values[0], bool):    
                cats = np.unique([str(x) for x in data])
            else:
                try:
                    cats = np.unique(data)
                except:
                    cats = np.unique([str(x) for x in data])
            imap = {}
            for i, cat in enumerate(cats):
                imap[cat] = i
            fun = lambda x: imap[x]
            if isinstance(data.values[0], bool):    
                return list(map(fun,[str(x) for x in data]))
            return list(map(fun, data))
        else:
            return data
    
    @staticmethod
    def is_categorical_series(data):
        """true if data is a categorical series"""
        return  isinstance(data.values[0], (str,bool))

class PlotDataFilter(Dashboard):
    
    def __init__(self,
                 data,
                 max=None,
                 min=None,
                 step=None,
                 low=None,
                 high=None,
                 description='plot data',
                 map_data = True,
                 default=None,
                 fixed_active=False,
                 **kwargs
                ):
        self.halt_update = True
        self._description = description
        self._data = data
        slicer = self._get_data_slicer(description=description)
        scaler = DataFrameScaler(slicer.output, max=max, min=min, step=step, low=low, high=high, name='data_scaler')
        self.output = pd.DataFrame(index=scaler.output.index, columns=[description])
        if max is None:
            max = 100
        if min is None:
            min = 0
        if high is None:
            high = 100
        if low is None:
            low = 0
        if step is None:
            step = 1
        if default is None:
            default = (high+low)*0.5
        if np.isnan(default):
            def_widget='HTML$N=default_w&v=0'
            self.default_value = np.nan
        else:
            def_widget = '@('+str(min)+','+str(max)+','+str(step)+','+str(default)+')$d=Default value'
        dash = ['r$N=main_row',
                [
                 slicer,
                 ['c$N=aply_col',[scaler,
                                  ['r$N=apply_row',['Map Data$N=map_text','@'+str(map_data)+'$N=map_chk','[['+str(map_data)+']]$N=map_valid']]
                                 ]],
                def_widget
                
                ]
                ]
        Dashboard.__init__(self, dash, **kwargs)
        
        if fixed_active:
            self.map_chk.visible = False
            self.map_valid.visible = False
            self.map_chk.target.disabled = True
            
        self.link('map_chk','map_valid')
        self.map_chk.target.layout.width = "100%"
        self.map_valid.target.readout = 'Mapping disabled'
        self.observe(self.update)
        self.halt_update = False
        self.update()
    
    @property
    def selector_vals(self):
        vals = self.data_slicer.selector_vals
        vals.update(self.data_scaler.selector_vals)
        vals['default_value'] = self.default_value.value
        vals['map_data'] = self.map_chk.value
        vals['default_min'] = self.default_value.target.min
        vals['default_max'] = self.default_value.target.max
        vals['default_step'] = self.default_value.target.step
        return vals
    
    def update_selectors(self,block=False,**kwargs):
        self.halt_update = True
        self.data_slicer.update_selectors(**kwargs)
        self.data_scaler.update_selectors(**kwargs)
        if 'max' in kwargs:
            kwargs['default_max'] = kwargs['max']
        if 'min' in kwargs:
            kwargs['default_min'] = kwargs['min']
        if 'step' in kwargs:
            kwargs['default_min'] = kwargs['min']  
        if 'default' in kwargs:
            kwargs['default_value'] = kwargs['default']
        
        if 'default_value' in kwargs:
            self.default_value.value = kwargs['default_value']
        if 'map_data' in kwargs:
            self.map_chk.value = kwargs['map_data']
            
        if 'default_min' in kwargs: 
            if kwargs['default_min'] <= self.default_value.value:
                self.default_value.target.min = kwargs['default_min']
            else:
                self.default_value.target.max = max(kwargs['default_min'],self.default_value.target.max)
                self.default_value.value =max(kwargs['default_min'],self.default_value.target.max)
                self.default_value.target.min = kwargs['default_min']
        if 'default_max' in kwargs:
            if kwargs['default_max'] >= self.default_value.value:
                self.default_value.target.max = kwargs['default_max']
            else:
                self.default_value.value = kwargs['default_max']
                self.default_value.target.max = kwargs['default_max']
        if 'default_step' in kwargs:
            self.default_value.target.step = kwargs['default_step']
        self.halt_update = False
        if not block:
            self.update()
        
    
    @property
    def data(self):
        return self._data
        
    @data.setter
    def data(self, val):
        self._data = val
        self.update()
        
    def _get_data_slicer(self, description):
        #leave room for other pandas structures
        if isinstance(self._data, pd.DataFrame):
            slicer = DataFrameSlicer(self._data, name='data_slicer', description=description)
            slicer.index_slicer.slice_mode.value = 'slice'
            return slicer
        
        elif isinstance(self._data, pd.Panel4D):
            slicer = Panel4DToPlot(self._data, name='data_slicer', description=description, output_mode='series')
            return slicer
        elif isinstance(self._data, pd.Panel):
            slicer = PanelToPlot(self._data, name='data_slicer', description=description)
            return slicer
        
    def update(self, _=None):
        if not self.halt_update:
            self.data_slicer.data = self._data
            self.data_scaler.data = self.data_slicer.output
            active_data = self.data_scaler.output
            if isinstance(active_data,pd.Series):
                columns = self.data_scaler.dd_sel.value
            else:
                columns = active_data.columns
            if isinstance(self._data, pd.DataFrame):
                empty_df = pd.DataFrame(index=self._data.index, columns=columns)
            else:
                empty_df = pd.DataFrame(index=self.data_scaler.output.index, columns=columns)
            if self.map_chk.value:
                if hasattr(self.default_value,'value'):
                    self.output = empty_df.combine_first(active_data).fillna(self.default_value.value)
                else:
                    self.output = empty_df.combine_first(active_data).fillna(self.default_value)
            else:
                if hasattr(self.default_value,'value'):
                    self.output = empty_df.fillna(self.default_value.value)
                else:
                    self.output = empty_df.fillna(self.default_value)
            self.output.columns = [self._description]

class PlotCmapFilter(Dashboard):
    
    def __init__(self,
                 data,
                 max=None,
                 min=None,
                 step=None,
                 low=None,
                 high=None,
                 description='plot_data',
                 map_data = True,
                 default_color='blue',
                 **kwargs
                ):
        self.halt_update = True
        self._data = data
        self._description = description
        slicer = self._get_data_slicer(description=description)
        scaler = DataFrameScaler(slicer.output, max=max, min=min, step=step, low=low, high=high, name='data_scaler')
        self.output = pd.DataFrame(index=scaler.output.index, columns=[description])
        cmap = ColormapPicker(name='cm_picker', mode='interactive')
        dash = ['r$N=main_row',
                [
                 slicer,
                 ['c$N=aply_col',[scaler,
                                  ['r$N=apply_row',['Map Data$N=map_text','@'+str(map_data)+'$N=map_chk','[['+str(map_data)+']]$N=map_valid']]
                                 ]],
                ['c$N=color_col',[ cmap,
                 '@cpicker$N=default_color&d=Default color&val='+default_color]]
                ]
                ]
        Dashboard.__init__(self, dash, **kwargs)
        self.link('map_chk','map_valid')
        self.map_chk.target.layout.width = "100%"
        self.map_valid.target.readout = 'Mapping disabled'
        self.observe(self.update)
        self.halt_update = False
        self.update()
    @property
    def selector_vals(self):
        vals = self.data_slicer.selector_vals
        vals.update(self.data_scaler.selector_vals)
        vals['default_color'] = self.default_color.value
        vals['cmap'] = dict(self.cm_picker.kwargs)
        vals['map_data'] = self.map_chk.value
        return vals
    
    def update_selectors(self,block=False,**kwargs):
        self.halt_update = True
        self.data_slicer.update_selectors(**kwargs)
        self.data_scaler.update_selectors(**kwargs)
        if 'default_color' in kwargs:
            self.default_color.value = kwargs['default_color']
        if 'cmap' in kwargs:
            self.cm_picker.set_kwargs(kwargs['cmap'])
        if 'map_data' in kwargs:
            self.map_chk.value = kwargs['map_data']
        self.halt_update = False
        if not block:
            self.update()
            #self.cm_picker.master_palette.update_masterpalette()
            self.cm_picker._on_close_click()
    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, val):
        self._data = val
        self.update()
    def _get_data_slicer(self, description):
        #leave room for other pandas structures
        if isinstance(self._data, pd.DataFrame):
            slicer = DataFrameSlicer(self._data, name='data_slicer', description=description)
            slicer.index_slicer.slice_mode.value = 'slice'
            return slicer
        
        elif isinstance(self._data, pd.Panel4D):
            slicer = Panel4DToPlot(self._data, name='data_slicer', description=description, output_mode='series')
            return slicer
        elif isinstance(self._data, pd.Panel):
            slicer = PanelToPlot(self._data, name='data_slicer', description=description)
            return slicer
    
    def update(self, _=None):
        if not self.halt_update:
            self.data_slicer.data = self._data
            self.data_scaler.data = self.data_slicer.output
            active_data = self.data_scaler.output.apply(lambda x: self.cm_picker.master_palette.map_data(x,hex=True))
            if isinstance(active_data,pd.Series):
                columns = self.data_scaler.dd_sel.value
            else:
                columns = active_data.columns

            if isinstance(self._data, pd.DataFrame):
                empty_df = pd.DataFrame(index=self._data.index, columns=columns)
            else:
                empty_df = pd.DataFrame(index=self.data_scaler.output.index, columns=columns)
            if self.map_chk.value:
                self.output = empty_df.combine_first(active_data).fillna(self.default_color.value)
            else:
                self.output = empty_df.fillna(self.default_color.value)
            self.output.columns = [self._description]

class PlotMapper(Dashboard):
    mapper_dict = {'x':{'max':100.0,
                    'min':0.0,
                    'step':0.1,
                    'high':1.,
                    'low':0.,
                    'default':np.nan,
                    'map_data':True,
                    'fixed_active':True,
                   },
               'y':{'max':100.0,
                    'min':0.0,
                    'step':0.1,
                    'high':1.,
                    'low':0.,
                    'default':np.nan,
                    'map_data':True,
                    'fixed_active':True,
                   },
               'size':{'max':100,
                    'min':1,
                    'step':0.5,
                    'high':20,
                    'low':10,
                    'default':12,
                    'map_data':False,
                    'fixed_active':False,
                   },
               'line_width':{'max':50,
                    'min':0,
                    'step':0.5,
                    'high':5,
                    'low':1,
                    'default':2,
                    'map_data':False,
                    'fixed_active':False,
                   },
               'fill_alpha':{'max':1.0,
                    'min':0.,
                    'step':0.05,
                    'high':0.95,
                    'low':0.3,
                    'default':1.,
                    'map_data':False,
                    'fixed_active':False,
                   },
               'line_alpha':{'max':1.0,
                    'min':0.,
                    'step':0.05,
                    'high':0.95,
                    'low':0.3,
                    'default':1.,
                    'map_data':False,
                    'fixed_active':False,
                   },
               'line_color':{'default_color':'black','map_data':False,'step':0.05,'min':0.0,'low':0.0},
               'fill_color':{'default_color':'blue','map_data':False,'step':0.05,'min':0.0,'low':0.0}
              }
    
    def __init__(self, data, mapper_dict=None, marker_opts=None, **kwargs):
        if marker_opts is None:
            self._marker_opts = ['circle', 'square', 'asterisk', 'circle_cross',
                                 'circle_x', 'square_cross', 'square_x', 'triangle',
                                 'diamond', 'cross', 'x', 'inverted_triangle',
                                 ]
        else:
            self._marker_opts = marker_opts
        self._data = data
        if isinstance(self._data, pd.DataFrame):
            if isinstance(self._data.index.values[0], (list,tuple,str)):
                old = self._data.index
                ix = [str(x) for x in self._data.index.values]
                self._data.index = ix
                self._data = self._data.reset_index()
                self._data.index = old
            else:
                ix = self._data.index
                self._data = self._data.reset_index()
                self._data.index = ix
        if mapper_dict is None:
            self.mapper_dict = self.get_default_mapper_dict()
        else:
            self.mapper_dict = mapper_dict
        self.params = list(sorted(self.mapper_dict.keys()))
        self.halt_update = True
        self.memory = {}
        cm = True
        da = True
        for p in self.params:
            if 'default_color' in self.mapper_dict[p].keys() and cm:
                cmap_filter = PlotCmapFilter(self._data, name='cmap_filter',description=p, mode='interactive', **self.mapper_dict[p])
                cm = False
                self._firstcolor = p
                self._color_keys = [p]
                self.memory[p] = cmap_filter.selector_vals
            elif da:
                data_filter = PlotDataFilter(self._data, name='data_filter', description=p, mode='interactive', **self.mapper_dict[p])
                da=False
                self._firstdata = p
                self.memory[p] = data_filter.selector_vals
            elif not da and not cm:
                break
        
        marker = Dashboard(['dd$d=Marker type&val=circle&o='+str(self._marker_opts)], name='marker' )
        
        dash = ['c$N=plot_mapper',['dd$d=Attribute&o='+str(self.params),marker,cmap_filter,data_filter]]
        Dashboard.__init__(self, dash, **kwargs)
        self._current = self.attribute.value
        self.init_data()
        self.observe(self.update)
        self.attribute.observe(self._update_dropdown)
        self.halt_update = False
        #for p in self.params:
        #    self.attribute.value = p
        self._update_dropdown()
        self.update()

    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, val):
        self._data = val
        self.data_filter.data = val
        self.cmap_filter.data = val
        self.update()

    @classmethod
    def get_default_mapper_dict(cls):
        return cls.mapper_dict
    
    def init_data(self):
        self.output = pd.DataFrame(index=self.data_filter.output.index,columns=self.params)
        self.output.loc[:,self.params[0]] = self.data_filter
        
        for param in self.params:
            kwargs = self.mapper_dict[param]
            if 'default_color' in kwargs.keys():
                self.cmap_filter.update_selectors(**kwargs)
                data  = self.cmap_filter.output.values.copy()  
                self.output.loc[:,param] = data
                self._color_keys.append(param)
                self.memory[param] = kwargs
            elif param != self._firstdata:
               
                self.data_filter.update_selectors(**kwargs)
                data = self.data_filter.output.values.copy()  
                self.output.loc[:,param] = data
                self.memory[param] = kwargs
    
    def _update_dropdown(self,_=None):
        p = self.attribute.value
        if p in self._color_keys:
            self.data_filter.visible = False
            self.cmap_filter.visible = True
            self.memory[self._current] =  self.cmap_filter.selector_vals
            self.cmap_filter.update_selectors(block=True,**self.memory[p])
        else:
            self.data_filter.visible = True
            self.cmap_filter.visible = False
            self.memory[self._current] =  self.data_filter.selector_vals
            self.data_filter.update_selectors(block=True,**self.memory[p])
    
    def update(self, _=None):
        param = self.attribute.value
        kwargs = self.mapper_dict[param]
        if 'default_color' in kwargs.keys():
            self.output.loc[:,param] = self.cmap_filter.output.values.copy()     
        else:
            self.output.loc[:,param] = self.data_filter.output.values.copy()  