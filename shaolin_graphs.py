import numpy as np

import pandas as pd
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import planarity

from shaolin.core.dashboard import Dashboard


class LayoutParams(Dashboard):
    """Widget for managing the parameters necessary for the layout calculator.
       It's able to manage all the default basic parameters that networkx takes as input
       
    """
    LAYOUT_PARAMS = {'dim':'both',
                 'scale':1,
                 'center':None,
                 'weigth':None,
                 'fixed':None,
                 'iterations':5000,
                 'prog':'neato',
                 'layout':'draw_graphviz',
                 'k':'exchange'}

    LAYOUTS = {'Circular layout': 'circular_layout',
               'Random layout': 'random_layout',
               'Shell layout':'shell_layout',
               'Spectral layout':'spectral_layout',
               'Spring layout':'spring_layout',
               'Graphviz layout':'draw_graphviz',
               'HCsoft layout':'dh_spring_layout'}
    
    @classmethod
    def default_layout_params(cls):
        return cls.LAYOUT_PARAMS
    @classmethod
    def available_layouts(cls):
        return cls.LAYOUTS
    
    def __init__(self, G, default_params=None,mode='interactive', **kwargs):
        if default_params is None:
            self.params = self.default_layout_params()
        else:
            self.params = default_params
        self._G = G
        dim_opts = {'2D':'2d',
                    '3D':'3d',
                    'Both':'both'}
        edges = G.edges()[0]
        ls_weights = list(G.edge[edges[0]][edges[1]].keys())
        if 'exchange' not in ls_weights:
            ls_weights += ['exchange']
        if 'weight' not in ls_weights:
            ls_weights += ['weight']
        dash = ['c$N=layout_params',
                [
                 '###Layout Params$N=layout_param_title',
                 ['r$N=body_row',['@sel$d=Layout&o='+str(self.available_layouts())\
                                  +'&val='+str(self.params['layout']),
                                  ['c$N=layout_row_1',
                                   [
                                    ['r$N=top_mini_box',
                                     ['@togs$d=Dimension&N=dim&o='+str(dim_opts)+'&val='+str(self.params['dim']),
                                     '@(1e-4,10e4,1,'+str(self.params['scale'])+')$d=Scale',
                                     'text$d=Center&N=center_display&val='+str(self.params['center'])
                                     ]
                                    ],
                                    ['r$N=bot_mini_box',['@selmul$d=Root node&N=fixed_display&o='+str(list(G.nodes())+['None']),
                                                         ['c$N=sub_box_layout',
                                                             ['@dd$d=Weight&o='+str(ls_weights+['None'])+'&val='+str(self.params['weigth']),
                                                              ['r$N=sub_subrow',['@(1,10e4,1,'+str(self.params['iterations'])+')$d=Iterations',
                                                                                 '@dd$N=kmet&d=k&o='+str(ls_weights)+'&val='+str(self.params['k']),
                                                                                 '@(0,1e4,1,0)$N=kfloat&d=k'
                                                                                ]
                                                              ],
                                                              '@togs$d=Prog&o=["neato", "twopi", "fdp", "dot", "circo"]&val='+str(self.params['prog'])
                                                             ]  
                                                         ]                                                                           
                                                        ]
                                    ]
                                  ]
                                 ]
                                ]
                 ]
                ]
               ]
        
        Dashboard.__init__(self, dash,mode=mode, **kwargs)
        self.fixed_display.value = [str(self.params['fixed'])]
        self.observe(self.update_display)
        self.observe(self.update_value)
        self.layout.observe(self.on_layout_change)
        self.update_display()
        #Can inherit a target when not used as a widget
        self.update_value()
        
    
    @property
    def G(self):
        return self._G
    @G.setter
    def G(self, val):
        self._G = val
        self.update()
    
    
    def update_display(self, _=None):
        """Updates the visual attributes of the widget"""
        simple = ['circular_layout', 'random_layout', 'shell_layout']
        if self.layout.value in simple:
            disable = ['prog', 'iterations', 'fixed_display',
                       'weight', 'kmet', 'kfloat']
            for item in disable:
                getattr(self, item).visible = False

        if self.layout.value == 'spectral_layout':
            disable = ['prog', 'iterations',
                       'fixed_display', 'kmet', 'kfloat']
            for item in disable:
                getattr(self, item).visible = False
            self.weight.visible = True

        if self.layout.value == 'spring_layout':
            disable = ['prog', 'kmet']
            for item in disable:
                getattr(self, item).visible = False
            self.weight.visible = True
            self.iterations.visible = True
            self.fixed_display.visible = True
            self.kfloat.visible = True

        if self.layout.value == 'draw_graphviz':
            disable = ['kmet', 'kfloat']
            for item in disable:
                getattr(self, item).visible = False
            self.weight.visible = True
            self.iterations.visible = True
            self.fixed_display.visible = True
            self.prog.visible = True

        if self.layout.value == 'dh_spring_layout':
            disable = ['prog', 'kfloat']
            for item in disable:
                getattr(self, item).visible = False
            self.fixed_display.visible = True
            self.weight.visible = True
            self.iterations.visible = True
            self.fixed_display.visible = True
            self.kmet.visible = True

    def single_val(self, dim=2, _=None):
        """Calculates the layout dict for a given dimension of the layout.
            dim: dimension of the layout we want to calculate its parameters.
            return: an attribute dictionary keyed with the name of the function
                    containing its kwargs
        """
        layout = {}
        simple = ['circular_layout', 'random_layout', 'shell_layout']
        if self.layout.value in simple:
            kwargs = {'dim':dim,
                      'scale':self.scale.value,
                      'center':self.get_center()}

        if self.layout.value == 'spectral_layout':
            kwargs = {'dim':dim,
                      'scale':self.scale.value,
                      'center':self.get_center(),
                      'weight':self.get_weight()}

        if self.layout.value == 'spring_layout':
            kwargs = {'dim':dim,
                      'scale':self.scale.value,
                      'center':self.get_center(),
                      'weight':self.get_weight(),
                      'iterations':int(self.iterations.value),
                      'fixed':self.get_fixed(),
                      'k':self.get_k()}

        if self.layout.value == 'draw_graphviz':
            kwargs = {'prog':self.prog.value,
                      'root':self.get_fixed()}

        if self.layout.value == 'dh_spring_layout':
            kwargs = {'dim':dim,
                      'scale':self.scale.value,
                      'center':self.get_center(),
                      'weight':self.get_weight(),
                      'iterations':int(self.iterations.value),
                      'fixed':self.get_fixed(),
                      'k':self.kmet.value}
        layout[self.layout.value] = kwargs
        return layout

    def get_fixed(self):
        """Gets the fixed value from the text input. it's a way to handle the None"""
        values = self.fixed_display.value
        if values[0] == 'None':
            return None
        else:
            return values

    def get_weight(self):
        """Gets the fixed value from the text input. it's a way to handle the None"""
        value = self.weight.value
        if value == 'None':
            return None
        else:
            return value

    def get_k(self):
        """Gets the k value from the text input. it's a way to handle the None"""
        if self.kfloat.value == 0:
            return None
        else:
            return int(self.kfloat.value)

    def get_center(self):
        """Gets the center value from the text input"""
        import ast
        if self.center_display.value == 'None':
            return None
        else:
            data = ast.literal_eval(self.center_display.value)
            np.array(data)

    def update_value(self, _=None):
        """Updates the target parameter dict for the selected dimensions"""
        dims = {}
        if self.dim.value == '2d':
            dims['2d'] = self.single_val(dim=2)
            dims['3d'] = None
        elif self.dim.value == '3d':
            dims['3d'] = self.single_val(dim=3)
            dims['2d'] = None
        elif self.dim.value == 'both':
            dims['2d'] = self.single_val(dim=2)
            dims['3d'] = self.single_val(dim=3)
        self.output = dims

    def on_value_change(self, _):
        """Trigger for value updater"""
        self.update_value()
    def on_layout_change(self, _):
        """Trigger for value and layout updater"""
        self.update_display()
        self.update_value()


class LayoutCalculator(Dashboard):
    """Calculates an arbitrary layout for a netxorkx Graph
    """

    def __init__(self, G,
                 default_params=None,
                 **kwargs
                ):
        if default_params is None:
            self.default_params = LayoutParams.default_layout_params()
        else:
            self.default_params = default_params
        self._G = G.copy()
        dash = ['c$N=layout_calculator',[LayoutParams(G, name='layout_params'),'btn$N=button&d=Run']]
        Dashboard.__init__(self, dash, **kwargs)
        self.button.observe(self.update)
        self.update()
        
    @property
    def G(self):
        return self._G
    @G.setter
    def G(self, val):
        self._G = val
        self.layout_params.G = val
    

    def update(self, G=None, _=None):
        """Starts the calculations from current LAYOUT_PARAMS"""                                    
        self.calculate_layouts()
        self.node = pd.Panel(self.node)
        self.edge = pd.Panel(self.edge)

    def one_layout(self, func, kwargs):
        """Calculates one arbitrary layout"""
        if 'fixed' in kwargs.keys():
            if not kwargs['fixed'] is None:
                kwargs['pos'] = nx.random_layout(self.G, dim=kwargs['dim'])
        if func == 'dh_spring_layout':
            return self.dh_spring_layout(self.G, **kwargs)
        elif func == 'draw_graphviz':
            return graphviz_layout(self.G, **kwargs)
        else:
            return getattr(nx, func)(self.G, **kwargs)

    def calculate_layouts(self):
        """Will calculate de layout according to its current LAYOUT_PARAMS configuration"""
        only_2d = ['circular_layout', 'shell_layout', 'draw_graphviz', 'graphviz_layout']
        self.node = {}
        self.edge = {}
        # both = self.params.dim.value not in self.params.target.keys()
        if True:#both:
            for sd in ['2d', '3d']:
                fun = list(self.layout_params.output[sd].keys())[0]
                kwargs = self.layout_params.output[sd][fun]
                layout = self.one_layout(fun, kwargs)
                df_ = pd.DataFrame(layout).T
                self.node[sd] = df_.copy()
                if sd == '2d':
                    if len(self.node[sd].columns) == 2:
                    #after the first one the df will have a 'z column filled with nan
                        df_.columns = ['x', 'y']
                        self.node[sd].columns = ['x', 'y']
                    edge_data = [(df_.x[e[0]], df_.y[e[0]], df_.x[e[1]], df_.y[e[1]])
                                 for e in self.G.edges_iter()]
                    edge_index = [e for e in self.G.edges_iter()]
                    self.edge[sd] = pd.DataFrame(index=edge_index,
                                                 data=edge_data,
                                                 columns=['x0', 'y0', 'x1', 'y1']
                                                )
                    self.edge[sd]['cx'] = (self.edge[sd]['x0']+self.edge[sd]['x1'])/2
                    self.edge[sd]['cy'] = (self.edge[sd]['y0']+self.edge[sd]['y1'])/2
                elif sd == '3d':
                    if fun in only_2d:
                        df_['z'] = 0
                        self.node[sd]['z'] = 0
                    if len(self.node[sd].columns) == 3:
                        df_.columns = ['x', 'y', 'z']
                        self.node[sd].columns = ['x', 'y', 'z']
                        
                    edge_data = [(df_.x[e[0]], df_.y[e[0]], df_.z[e[0]],
                                  df_.x[e[1]], df_.y[e[1]], df_.z[e[1]])
                                 for e in self.G.edges_iter()
                                ]
                    edge_index = [e for e in self.G.edges_iter()]
                    self.edge[sd] = pd.DataFrame(index=edge_index,
                                                 data=edge_data,
                                                 columns=['x0', 'y0', 'z0',
                                                          'x1', 'y1', 'z1']
                                                )
                    self.edge[sd]['cx'] = (self.edge[sd]['x0']+self.edge[sd]['x1'])/2
                    self.edge[sd]['cy'] = (self.edge[sd]['y0']+self.edge[sd]['y1'])/2
                    self.edge[sd]['cz'] = (self.edge[sd]['z0']+self.edge[sd]['z1'])/2
        else:#TODO: handle single layout manipulation and arbitrary dimension layoyt
            return

    def dh_spring_layout(self, G,
                         dim=2,
                         k=None,
                         pos=None,
                         fixed=None,
                         iterations=50,
                         weight='weight',
                         scale=1.0,
                         center=None,
                         kname='var'
                        ):
        """Position nodes using a modified spring layout that allows for different k for each node.
           Same interface as nx.spring_layout.
        Parameters
        ----------
        G : NetworkX graph or list of nodes
        dim : int
           Dimension of layout
        k : array (default=None)
           Optimal distance between nodes.  If you want it to make it None
           just use nx.spring_layout.
        pos : dict or None  optional (default=None)
           Initial positions for nodes as a dictionary with node as keys
           and values as a list or tuple.  If None, then use random initial
           positions.
        fixed : list or None  optional (default=None)
          Nodes to keep fixed at initial position.
        iterations : int  optional (default=50)
           Number of iterations of spring-force relaxation
        weight : string or None   optional (default='weight')
            The edge attribute that holds the numerical value used for
            the edge weight.  If None, then all edge weights are 1.
        scale : float (default=1.0)
            Scale factor for positions. The nodes are positioned
            in a box of size [0,scale] x [0,scale].
        center : array-like or None
           Coordinate pair around which to center the layout.
        Returns
        -------
        dict :
           A dictionary of positions keyed by node
        Examples
        --------
        >>> G=nx.path_graph(4)
        >>> pos=nx.spring_layout(G)
        # The same using longer function name
        >>> pos=nx.fruchterman_reingold_layout(G)
        """
        import numpy as np
        def _rescale_layout(pos, scale=1):
            # rescale to (-scale,scale) in all axes
            # shift origin to (0,0)
            lim = 0 # max coordinate for all axes
            for i in range(pos.shape[1]):
                pos[:, i] -= pos[:, i].mean()
                lim = max(pos[:, i].max(), lim)
            # rescale to (-scale,scale) in all directions, preserves aspect
            for i in range(pos.shape[1]):
                pos[:, i] *= scale/lim
            return pos

        def process_params(G, center, dim):
            """ Some boilerplate code."""
            import numpy as np

            if not isinstance(G, nx.Graph):
                empty_graph = nx.Graph()
                empty_graph.add_nodes_from(G)
                G = empty_graph

            if center is None:
                center = np.zeros(dim)
            else:
                center = np.asarray(center)

            if len(center) != dim:
                msg = "length of center coordinates must match dimension of layout"
                raise ValueError(msg)

            return G, center

        G, center = process_params(G, center, dim)

        if fixed is not None:
            nfixed = dict(zip(G, range(len(G))))
            fixed = np.asarray([nfixed[v] for v in fixed])

        if pos is not None:
            # Determine size of existing domain to adjust initial positions
            dom_size = max(np.array(list(pos.values())).flatten())
            shape = (len(G), dim)
            pos_arr = np.random.random(shape) * dom_size + center
            for i, n in enumerate(G):
                if n in pos:
                    pos_arr[i] = np.asarray(pos[n])
        else:
            pos_arr = None

        if len(G) == 0:
            return {}
        if len(G) == 1:
            return {G.nodes()[0]: center}

        try:
            # Sparse matrix
            if len(G) < 500:  # sparse solver for large graphs
                raise ValueError
            A = nx.to_scipy_sparse_matrix(G, weight=weight, dtype='f')
            k = nx.to_scipy_sparse_matrix(G, weight=kname, dtype='f')
            if k is None and fixed is not None:
                # We must adjust k by domain size for layouts that are not near 1x1
                nnodes, _ = A.shape
                k = dom_size / np.sqrt(nnodes)
            pos = _sparse_fruchterman_reingold(A, dim, k, pos_arr, fixed, iterations)
        except:
            A = nx.to_numpy_matrix(G, weight=weight)
            k = nx.to_numpy_matrix(G, weight=kname)
            k = np.asarray(k)
            if k is None and fixed is not None:
                # We must adjust k by domain size for layouts that are not near 1x1
                nnodes, _ = A.shape
                k = dom_size / np.sqrt(nnodes)
            pos = self._fruchterman_reingold(A, dim, k, pos_arr, fixed, iterations)
        if fixed is None:
            pos = _rescale_layout(pos, scale=scale) + center
        pos = dict(zip(G, pos))
        return pos


    def _fruchterman_reingold(self, A, dim=2, k=None, pos=None, fixed=None,
                              iterations=50):
                                  
        def __magics(x,top=1e8,low=1e-18):
            if  x<1:
                val = np.sqrt(1-x)/(1-x)
                return val
            #elif x<=low:
            #    return 0
            else: 
                return top
                
        def __force(M0,dist):
           
            M = pd.DataFrame(dist)
            force = (M.applymap(lambda x: __magics(x)).values-M0)/M0
            return force
        # Position nodes in adjacency matrix A using Fruchterman-Reingold
        # Entry point for NetworkX graph is fruchterman_reingold_layout()
        try:
            import numpy as np
        except ImportError:
            raise ImportError("_fruchterman_reingold() requires numpy: http://scipy.org/ ")

        try:
            nnodes, _ = A.shape
        except AttributeError:
            raise nx.NetworkXError(
                "fruchterman_reingold() takes an adjacency matrix as input")

        A = np.asarray(A) # make sure we have an array instead of a matrix

        if pos is None:
            # random initial positions
            pos = np.asarray(np.random.random((nnodes, dim)), dtype=A.dtype)
        else:
            # make sure positions are of same type as matrix
            pos = pos.astype(A.dtype)

        # optimal distance between nodes
        if k is None:
            k = np.sqrt(1.0/nnodes)

        # the initial "temperature"  is about .1 of domain area (=1x1)
        # this is the largest step allowed in the dynamics.
        # We need to calculate this in case our fixed positions force our domain
        # to be much bigger than 1x1

        #pos.T[0] is the first coordinate of the nnode vector
        t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1]))*0.1
        # simple cooling scheme.
        # linearly step down by dt on each iteration so last iteration is size dt.
        dt = t/float(iterations+1)
        delta = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1]), dtype=A.dtype)
        umatrix = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1]), dtype=A.dtype)
        # the inscrutable (but fast) version
        # this is still O(V^2)
        # could use multilevel methods to speed this up significantly
        for iteration in range(iterations):
            # matrix of difference between points
            for i in range(pos.shape[1]):
                delta[:, :, i] = pos[:, i, None]-pos[:, i]
            # distance between points
            distance = np.sqrt((delta**2).sum(axis=-1))
            # enforce minimum distance of 0.01
            distance = np.where(distance < 0.01, 0.01, distance)
            A = np.where(A < 0.001, 0.001, A)

            for i in range(pos.shape[1]):
                umatrix[:, :, i] = delta[:, :, i]/distance
            # displacement "force"
            #displacement = np.transpose(np.transpose(umatrix)*k*(distance-A)/A).sum(axis=1)
            displacement = np.transpose(np.transpose(umatrix)*k*__force(A,distance)).sum(axis=1)
            
            # update positions
            length = np.sqrt((displacement**2).sum(axis=1))
            length = np.where(length < 0.01, 0.1, length)
            delta_pos = np.transpose(np.transpose(displacement)*t/length)
            if fixed is not None:
                # don't change positions of fixed nodes
                delta_pos[fixed] = 0.0
            pos += delta_pos
            # cool temperature
            t -= dt
        return pos

class MatrixParams(Dashboard):
    """Thi widget handles the parameter for creating the matrix we
    will transform into a graph:
    """
    MATRIX_PARAMS = {'norm_min': 0,
                     'norm_max': 1,
                     'clip_min':1e-5,
                     'clip_max':1e10,
                     'target':[('clip', (1e-5,1e10))],#f[(fun,(kwargs))]
                     }

    TARGET_OPTIONS = {'Raw':'raw', #Do nothing
                      'Clip':'clip', #limit the matrix values betwen (min,max)
                      'Scale':'scale', #scale the matrix values between (min,max)
                      'Zscore':'zscore', #zscore the matrix elements
                      'Distance':'compute_distance', #Convert corr into a distance
                      }

    def __init__(self, default_params=None, **kwargs):
        if default_params is None:
            self.params = self.default_matrix_params()
        else:
            self.params = default_params
        
        dash = ['c$N=matrix_params',
                ['###Matrix params$N=matparams_title',
                 ['r$N=body_box',[
                        ['c$N=mp_num_box',['@(-50.,50.,1.,'+str(float(self.params['norm_min']))+')$N=norm_min&d=Rescale min',
                                           '@(-50.,50.,1.,'+str(float(self.params['norm_max']))+')$N=norm_max&d=Rescale max',
                                           '@[-5e5,2e10,1.,'+str(float(self.params['clip_min']))+']$N=clip_min&d=Clip min',
                                           '@[-5e5,2e10,1.,'+str(float(self.params['clip_max']))+']$N=clip_max&d=Clip max'
                                          ]
                        ],
                        ['c$N=mp_target_box',[['@sel$N=target_selector&d=Transform matrix&o='+str(self.get_target_options())+'&val='+str(self.params['target'][0][0])],
                                              ['r$N=buttonbox',['btn$d=Add','btn$d=Delete']],
                                              ['@texta$N=target_display&val='+str(self.params['target'])]
                                              
                                             ]
                        ]
                 ]
                 ]
                ]
               ]
        
        
                    
        Dashboard.__init__(self, dash, **kwargs)
        self.add.observe(self.on_add_clicked)
        self.delete.observe(self.on_del_clicked)
        self.observe(self.update)
        self.update()

    @classmethod
    def default_matrix_params(cls):
        return cls.MATRIX_PARAMS

    @classmethod
    def get_target_options(cls):
        return cls.TARGET_OPTIONS

    def update(self, _=None):
        """Updates the numeric MATRIX_ATTRIBUTES: max and min values for clip and scale"""
        self.transf_params = {'raw':None,
                                  'clip':(self.clip_min.value,
                                          self.clip_max.value),
                                  'scale':(self.norm_min.value,
                                           self.norm_max.value),
                                  'zscore':None,
                                  'compute_distance':None,
                                 }
        self.output = self.params['target']

    def on_del_clicked(self, _):
        """Trigger for updating the center widget.from del button."""
        self.params['target'].pop()
        self.target_display.value = str(self.params['target'])

    def on_add_clicked(self, _):
        """Trigger for updating the center widget.from del button."""
        key = self.target_selector.value
        new = (key, self.transf_params[key])
        self.params['target'].append(new)
        self.target_display.value = str(self.params['target'])

class GraphParams(Dashboard):
    """Widget for managing the parameters needed to convert a matrix into a graph
        Params:
        ---------------------------------------------------------------------
    """
    GRAPH_PARAMS = {'graph_type':'full',
                   'inverted' :  False,
                   'threshold': 0.00,
                   'target_attr':'exchange'}

    NODE_METRICS = ['betweenness_centrality', 'betweenness', 'degree','degree_centrality', 'degree_weighted',
                    'eigenvector_weighted', 'eigenvector', 'closeness_weighted',
                    'closeness', 'eccentricity_weighted', 'eccentricity']

    NODE_TARGET_ATTRS = {'Simetric max':'M',
                         'Correlation':'corr',
                         'Covariance': 'cov',
                         'Market custom': 'exchange',
                         'k spring': 'k_price',
                         'L0': 'l0',
                         'Simetric min' : 'm'}
    @classmethod
    def default_graph_params(cls):
        return cls.GRAPH_PARAMS
    
    @classmethod
    def default_node_metrics(cls):
        return cls.NODE_METRICS
    
    @classmethod
    def default_target_attributes(cls):
        return cls.NODE_TARGET_ATTRS
    
    def __init__(self, default_params=None,
                 metrics=None,
                 target=None,
                **kwargs):
        if default_params is None:
             params = self.default_graph_params()
        else:
            params = default_params
        if metrics is None:
            metrics = self.default_node_metrics()
        if target is None:
            dict_target_attrs = self.default_target_attributes()
        else:
            dict_target_attrs = target
            
        options = {'Mst':'mst',
                   'Pmfg':'pmfg',
                   'Full Matrix':'full'}
        dash = ['c$N=graph_params',['###Graph params$N=gaphparams_title',
                                  '@togs$d=Graph type&o='+str(options)+'&val='+str(params['graph_type']),
                                  ['r$N=gparams_row',['@[0.,50e10,1,'+str(params['threshold'])+']$N=threshold&d=Abs val threshold',
                                                    '@False$N=inverted&d=Invert distance'
                                                   ]
                                  ],
                                  '@selmul$d=Metrics&o='+str(metrics),
                                  '@dd$N=target_attr&d=Target attribute&o='+str(dict_target_attrs)+'&val='+str(params['target_attr'])
                                 ]
               ]
        Dashboard.__init__(self, dash, **kwargs)
        self.metrics.value = tuple(metrics)



class GraphMaker(object):
    """Transforms a matrix panel into a networkx Graph.
       Parameters
       ----------
       matrix_panel: pd.Panel.-Items axis:list of edge metrics
                              -Major and minor axis: list of node labels
       node_data: pd.Dataframe: Index: Node labels.
                               columns: node metrics.
       threshold: Minimum absolute value for node matrix items.
       matrix_params: see  selectors.wranglers.MatrixParams
       graph_type: kind of transformation in ('full', 'mst', 'pmfg'])
      """
    def __init__(self, matrix_panel,
                 node_data,
                 inv=False,
                 threshold=0.0,
                 matrix_params=[('raw', None)],
                 graph_type='full',
                 target='corr'):

        self.params = matrix_params
        self.panel = matrix_panel
        data = matrix_panel[target].copy()
        self.df_distance = self.calculate_graph_matrix(data,
                                                       matrix_params)
        self.G = self.create_graph(node_data,
                                   graph_type=graph_type,
                                   inv=inv,
                                   threshold=threshold)

    def add_node_info(self, g, key, pds):
        """Inserts the info from node_data into node attributes of the created graph
        """
        new_attributes = map(lambda x: (x[0], x[1]),
                             filter(lambda i: g.has_node(i[0]),
                                    pds.iteritems()
                                   )
                            )
        nx.set_node_attributes(g, key, dict(new_attributes))

    def raw(self, data, attrs=None):
        """Do nothing to the graph matrix"""
        return data

    def zscore(self, data, attrs=None):
        """Zscore the graph matrix"""
        return (data-np.mean(data))/data.std()

    def scale(self, data, attrs):
        """Scale the matrix. Attrs:(min,max)"""
        m, M = attrs
        return ((M-m)*(data-data.min())) / (data.max()-data.min()) + m

    def compute_distance(self, correlation_matrix, attrs):
        """Converts correlation into a distance"""
        return np.sqrt(2 * np.clip(1. - correlation_matrix ** 2, 0., 2.))

    def clip(self, data, attrs):
        """Limit a given matrix between (min,max)"""
        m, M = attrs
        return np.clip(data, m, M)

    def calculate_graph_matrix(self, data, matrix_params):
        """Convert the input matrix into the graph matrix
        applying an arbitrary set of transforms
        """
        for name, attrs in matrix_params:
            func = getattr(self, name)
            data = func(data, attrs)
        return data

    def create_graph(self,
                     data,
                     graph_type='mst',
                     inv=False,
                     threshold=0.0):
        """Converts the calculated graph adjacency matrix into a networkx Graph
        """
        df_distance = self.df_distance
        matrix_panel = self.panel
        if graph_type == 'mst':
            G = self.construct_mst(df_distance, matrix_panel,
                                   inv=inv, threshold=threshold)
            for col in data.columns:
                self.add_node_info(G, col, data[col])
        elif graph_type == 'pmfg':
            G = self.construct_pmfg(df_distance, matrix_panel,
                                    inv=inv, threshold=threshold)
            for col in data.columns:
                self.add_node_info(G, col, data[col])
        else:
            G = self.construct_full_graph(df_distance, matrix_panel,
                                          inv=inv, threshold=threshold)
            for col in data.columns:
                self.add_node_info(G, col, data[col])
        return G

    def construct_full_graph(self, df_distance, matrix_panel, inv=False, threshold=0.0):
        """Creates a graph with the provided adjacency
            matrix inserting all the node and edge metrics into the graph
        """
        if inv:
            df_distance = 1.0/df_distance

        g = nx.Graph()
        names = df_distance.columns.unique()
        for name in names:
            g.add_node(name)

        for i, n in enumerate(names):
            for j, m in enumerate(names):
                if j >= i:
                    break
                val = df_distance.loc[n, m]
                if np.abs(val) > threshold / len(names):
                    edgedict = {'weight': float(val)}
                    for item in matrix_panel.items:
                        edgedict[item] = float(matrix_panel.ix[item, n, m])
                    g.add_edge(n, m, edgedict)
        labels = {}
        for i, n in enumerate(names):
            name = names[i]
            labels[i] = name
        g = nx.relabel_nodes(g, labels)
        return g

    def construct_mst(self, df_distance, matrix_panel, inv=False, threshold=0.0):
        """Trim the provided adjacency matrix to create a Minimum Spanning Tree
            matrix inserting all the node and edge metrics into the graph
        """
        if inv:
            df_distance = 1.0/df_distance
        g = nx.Graph()
        names = df_distance.columns.unique()
        for name in names:
            g.add_node(name)
        for i, n in enumerate(names):
            for j, m in enumerate(names):
                if j >= i:
                    break
                val = df_distance.loc[n, m]
                # Bonferroni correction: TODO check implementation
                if np.abs(val) > threshold / len(names):
                    edgedict = {'weight': float(val)}
                    for item in matrix_panel.items:
                        edgedict[item] = float(matrix_panel.ix[item, n, m])
                    g.add_edge(n, m, edgedict)
        return nx.minimum_spanning_tree(g)

    def construct_pmfg(self, df_distance, matrix_panel, inv=False, threshold=0.0):
        """Trim the provided adjacency matrix to create a Planary Maximum Filtered Graph
            matrix inserting all the node and edge metrics into the graph
        """
        if inv:
            dist_matrix = 1.0/df_distance.values
        else:
            dist_matrix = df_distance.values
        index_upper_triangular = np.triu_indices(dist_matrix.shape[0], 1)
        isort = np.argsort(dist_matrix[index_upper_triangular])
        G = nx.Graph()
        names = df_distance.columns.unique()

        for k in np.arange(0, len(isort)):
            u = index_upper_triangular[0][isort[k]]
            v = index_upper_triangular[1][isort[k]]
            if np.abs(dist_matrix[u, v]) > threshold / len(names): # optional bonferroni correction
                edgedict = {'weight': float(dist_matrix[u, v])}
                for item in matrix_panel.items:
                    edgedict[item] = float(matrix_panel.ix[item, u, v])
                G.add_edge(u, v, edgedict)
                if not planarity.is_planar(G):
                    G.remove_edge(u, v)
        labels = {}
        for i, n in enumerate(names):
            name = names[i]
            labels[i] = name
        G = nx.relabel_nodes(G, labels)
        return G



from shaolin.dashboards.bokeh import *

from shaolin_experimental import *


class CentralityComputer(object):
    """Based on the github repo of Miguel Vaz https://github.com/mvaz/
        This Class computes the node metrics available in netwokrx, ranks them and
        stores the metrics in a DataFrame
    """
    NODE_METRICS = ['betweenness_centrality',
                'betweenness',
                'degree',
                'degree_centrality',
                'degree_weighted',
                'eigenvector_weighted',
                'eigenvector',
                'closeness_weighted',
                'closeness',
                'eccentricity_weighted',
                'eccentricity']
    @classmethod
    def default_node_metrics(cls):
        return cls.NODE_METRICS
    
    def __init__(self, G, metrics=None, custom=None):
        if metrics is None:
            self.ls_metrics = self.default_node_metrics()
        else:
            self.ls_metrics = metrics
        self.metrics = self.compute_metrics(G)

    def compute_metrics(self, g):
        """computes a given metric from a list of target metrics"""
        c = self.compute_one(g, self.ls_metrics[0])#first metrics must not be dependent
        
        #c[self.ls_metrics[0]] = met0[self.ls_metrics[0]]
        #c[self.ls_metrics[0]+'_rank'] = met0[self.ls_metrics[0]+'_rank']
        for target in self.ls_metrics:
            _metric = self.compute_one(g, target)
            try:
                _cs = _metric.columns.values
                c[_cs[0]] = _metric[_cs[0]]
                c[_cs[1]]= _metric[_cs[1]]
            except:
                pass
        return c.T

    def compute_one(self, g, target):
        """##How would yo like to define your custom metrics? just by passing
        #a function as parameter or something like custom={'Function name':fun}
        #or something different, i dont know wich one is better
        #quick example:
        #I you just set the attibute Y externally
        #self.Y = None
        #self.custom_params
        #elif target == 'Y':
        #    return self.Y(g,**self.custom_params)"""
        if target == 'betweenness_centrality':
            try:
                return self.betweenness_centrality(g, weighted=True)
            except:
                return np.nan
        elif target == 'betweenness':
            try:
                return self.betweenness_centrality(g, weighted=False)
            except:
                return np.nan
        elif target == 'degree_centrality':
            try:
                return self.degree_centrality(g, weighted=False)
            except:
                return np.nan
        elif target == 'degree':
            try:
                return self.degree(g, weighted=False)
            except:
                return np.nan
        elif target == 'degree_weighted':
            try:
                return self.degree_centrality(g, weighted=True)
            except:
                return np.nan
        elif target == 'eigenvector_weighted':
            try:
                return self.eigenvector_centrality(g)
            except:
                return np.nan
        elif target == 'eigenvector':
            try:
                return self.eigenvector_centrality(g, weighted=False)
            except:
                return np.nan
        elif target == ' closeness_weighted':
            try:
                return self.closeness(g)
            except:
                return np.nan
        elif target == 'closeness':
            try:
                return self.closeness(g, weighted=False)
            except:
                return np.nan
        elif target == 'eccentricity_weighted':
            try:
                return self.eccentricity(g)
            except:
                return np.nan
        elif target == 'eccentricity':
            try:
                return self.eccentricity(g, weighted=False)
            except:
                return np.nan


    def calc_metric(self, metric, metric_name, g, **kwargs):
        """Calculates an arbitrary metric"""
        return pd.DataFrame.from_dict(metric(g, **kwargs),
                                      orient='index').rename(columns={0: metric_name})


    #def compute_rank(self,col, order='ascending'):
    #    return col.rank(ascending=(order == 'ascending'))


    def compute_rank(self, col, order='ascending'):
        """Ranks the metrics"""
        u, v = np.unique(col if order == 'ascending' else -col, return_inverse=True)
        return (np.cumsum(np.bincount(v, minlength=u.size)) - 1)[v]


    def betweenness_centrality(self, g, weighted=True):
        """computes the given metric and returns it as a series"""
        _name = 'betweenness' + ('_weighted' if weighted else '')
        _d = self.calc_metric(nx.algorithms.centrality.betweenness_centrality,
                              _name, g, weight='weight' if weighted else None)
        _d[_name+'_rank'] = self.compute_rank(_d[_name], order='descending')
        return _d


    def degree_centrality(self, g, weighted=True):
        """computes the given metric and returns it as a series"""
        # TODO weight not yet taken into account, make own function
        _name = 'degree_cent' + ('_weighted' if weighted else '')
        _d = self.calc_metric(nx.algorithms.centrality.degree_centrality,
                              _name, g)  # , weight='weight' if weighted else None)
        
        _d[_name+'_rank'] = self.compute_rank(_d[_name], order='descending')
        return _d
    
    def degree(self, g, weighted=False):
        """computes the given metric and returns it as a series"""
        # TODO weight not yet taken into account, make own function
        _name = 'degree' + ('_weighted' if weighted else '')
        _d = self.calc_metric(nx.degree,
                              _name, g)  # , weight='weight' if weighted else None)
        
        _d[_name+'_rank'] = self.compute_rank(_d[_name], order='descending')
        return _d


    def eigenvector_centrality(self, g, weighted=True):
        """computes the given metric and returns it as a series"""
        _name = 'eigenvector' + ('_weighted' if weighted else '')
        _d = self.calc_metric(nx.algorithms.centrality.eigenvector_centrality_numpy, _name, g,
                              weight='weight' if weighted else None)
        _d[_name+'_rank'] = self.compute_rank(_d[_name], order='descending')
        return _d


    def eccentricity(self, g, weighted=True):
        """computes the given metric and returns it as a series"""
        ### weight not yet taken into accoutn
        _name = 'eccentricity' + ('_weighted' if weighted else '')
        _d = self.calc_metric(nx.algorithms.distance_measures.eccentricity,
                              _name, g)  # , weight='weight' if weighted else None)
        _d[_name+'_rank'] = self.compute_rank(_d[_name], order='ascending')
        return _d


    def closeness(self, g, weighted=True):
        """computes the given metric and returns it as a series"""
        def calc2(metric, metric_name, g, distance='weight', **kwargs):
            return pd.DataFrame.from_dict(metric(g, distance=distance),
                                          orient='index').rename(columns={0: metric_name})

        _name = 'closeness' + ('_weighted' if weighted else '')
        _d = self.calc_metric(nx.algorithms.centrality.closeness.closeness_centrality, _name, g,
                              distance='weight' if weighted else None)
        _d[_name+'_rank'] = self.compute_rank(_d[_name], order='ascending')
        return _d

class GraphCalculator(Dashboard):
    """Handles everything to do with graph managing.
       Attributes:
       ----------
       matrix_panel: pd.Panel of edge metrics/data
       gp: GraphParams object used to feed its graph parameters
       mp: MatrixParams object used to manage its graph's adjacency matrix
       gm: GraphMaker object in charge of graphs related calculations
       widget: GUI for selecting all the params related to graph calculations
    """
    def __init__(self, G,
                 mode='interactive',
                 metrics=False,
                 **kwargs):
        self._directed = G.is_directed()
        self.G = G.to_directed()
        
        self.metrics = metrics
        dash = ['c$N=graphcalculator',
                ['###Graph Calculator$N=gc_title',
                 ['r$N=body_row',[
                                  LayoutCalculator(self.G, name='layout', mode='interactive')
                                 ]
                 ],
                 ['r$N=buttons_row',['btn$d=Calculate',
                                     'tog$N=toggle_gp&d=Creation',
                                     'tog$N=toggle_mp&d=Matrix',
                                     'tog$N=toggle_layout&d=Layout',
                                     'HTML$N=display'
                                    ]
                 ]
                ]
               ]
        
        Dashboard.__init__(self, dash, **kwargs)
        self.display.value = self.display_str('Select and click Calculate')
        
    
        

        self.calculate.observe(self.on_calculate_click)
        self.toggle_layout.observe(self.on_toglay_change)
        self.update()
        self.toggle_layout.value = True
        

    def on_togmp_change(self, change=None):
        self.mp.visible = self.toggle_mp.value

    def on_toggp_change(self, change=None):
        self.gp.visible = self.toggle_gp.value

    def on_toglay_change(self, change=None):
        self.layout.visible = self.toggle_layout.value
        self.layout.button.visible = False

    def display_str(self, s):
        html = '''<div class="graph-gc"
                     style=" font-size:18px;
                            font-weight: bold; 
                            text-align:right;">'''+s+'''</div>'''
        return html

    def on_calculate_click(self, b):
        self.display.value = self.display_str('Calculating...')
        self.update()
        self.display.value = self.display_str('Ready. You can select again.')

    def update(self, _=None):
        def mdg_to_edge_df(g):
            edges = g.edges_iter()
            u0,v0 = next(edges)
            fi_data = g.edge[u0][v0]
            if not  isinstance(g,(nx.MultiDiGraph,nx.MultiGraph)):
                data_cols = list(fi_data.keys())
            else:
                data_cols = list(list(fi_data.values())[0].keys())
            cols = data_cols + ['src','dst']
            df = pd.DataFrame(index=np.arange(len(g.edges())),columns=cols)
            df.loc[:,['src','dst']] = np.array(g.edges())
            for c in data_cols:
                if not  isinstance(g,(nx.MultiDiGraph,nx.MultiGraph)):
                    df.loc[0,c] = fi_data[c]
                else:
                    df.loc[0,c] = fi_data[list(fi_data.keys())[0]][c]
            i = 1
            for u,v in edges:
                e_data = g.edge[u][v]
                for c in data_cols:
                    if not  isinstance(g,(nx.MultiDiGraph,nx.MultiGraph)):
                        df.loc[i,c] = e_data[c]
                    else:
                        df.loc[i,c] = e_data[list(fi_data.keys())[0]][c]
                i += 1
            return df
        if self.metrics and not  isinstance(self.G,(nx.MultiDiGraph,nx.MultiGraph)):
            if self._directed:
                self.comp = CentralityComputer(self.G)
            else:
                self.comp = CentralityComputer(self.G.to_undirected())
            self.G, self.node = self.add_graph_node_attributes(self.G,
                                                                   self.comp.metrics.T.copy())                                           
        else:
            self.node = pd.DataFrame(self.G.node).T
        self.node = self.node.dropna(axis=1,how='all')
        self.node.loc[:, 'label'] = self.node.index.values
        self.node.loc[:, 'index'] = np.arange(len(self.node.index.values))
        if not  isinstance(self.G,(nx.MultiDiGraph,nx.MultiGraph)):
            self.G = self.add_graph_edge_attributes(self.G)
        #self.edge = pd.Panel(self.G.edge).swapaxes(0, 1)
        self.edge = mdg_to_edge_df(self.G)
        self.layout.update(self.G)

    def add_node_info(self, G, key, pds):
        new_attributes = map(lambda x: (x[0], x[1]), filter(lambda i: G.has_node(i[0]),
                                                            pds.iteritems())
                            )
        nx.set_node_attributes(G, key, dict(new_attributes))

    def add_edge_info(self, G, key, data):
        for i, E in enumerate(G.edges()):
            u, v = E
            G.edge[u][v][key] = data.iloc[i]

    def add_graph_edge_attributes(self, G, func_list=None):
        if func_list is None:
            func_list = [nx.edge_betweenness_centrality, nx.edge_betweenness,
                         nx.edge_current_flow_betweenness_centrality]#,nx.eigenvector_centrality]
        columns = [f.__name__ for f in func_list]
        dft = pd.DataFrame(index=np.arange(len(G.edges())), columns=columns).sort_index()
        for i, name in enumerate(columns):
            try:
                vals = func_list[i](G)
            except:
                continue
            values = list(vals.values())
            dft[name] = values
        for name in columns:
            self.add_edge_info(G, name, dft[name])
        return G

    def add_graph_node_attributes(self, G, metrics):
        for name in metrics.columns:
            self.add_node_info(G, name, metrics[name])
        return G, pd.DataFrame(G.node).T



class GraphPlotBokeh(ToggleMenu):
    
    def __init__(self, gc, name='scatter_plot', **kwargs):
        self._width = 1200
        self._height = 800
        node_mapper_dict = {'size':{'max':100,
                            'min':1,
                            'step':0.5,
                            'high':75,
                            'low':25,
                            'default':40,
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
                       'fill_color':{'default_color':'#FFFFFF','map_data':False,'step':0.05,'min':0.0,'low':0.0}
                      }
        edge_mapper_dict = {'size':{'max':100,
                            'min':1,
                            'step':0.5,
                            'high':20,
                            'low':7,
                            'default':2,
                            'map_data':False,
                            'fixed_active':False,
                           },
                       'line_width':{'max':50,
                            'min':0,
                            'step':0.5,
                            'high':5,
                            'low':1,
                            'default':3,
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
                       'fill_color':{'default_color':'#EDB021','map_data':False,'step':0.05,'min':0.0,'low':0.0}
                      }
        gc.name = 'gc'
        output_notebook(hide_banner=True)
        edge_mapper_data = gc.edge
        node_mapper = PlotMapper(gc.node, mapper_dict=node_mapper_dict, name='node_mapper', mode='interactive')
        #node_mapper.marker.marker_type.value = 'square'
        edge_mapper = PlotMapper(edge_mapper_data, mapper_dict=edge_mapper_dict, name='edge_mapper', mode='interactive')
        node_tooltip_data = gc.node
        node_tooltip = BokehDataFrameTooltip(node_tooltip_data, name='node_tooltip')
        edge_tooltip = BokehDataFrameTooltip(edge_mapper_data, name='edge_tooltip')
        
        ToggleMenu.__init__(self,
                            children=[Dashboard(["r$n=buton_dash",["btn$d=Update&n=update_btn",'tog$d=Plot text']],name='plot_ctrl'),gc,node_mapper,
                                            edge_mapper,
                                            node_tooltip,
                                            edge_tooltip],
                            name=name,**kwargs)
        #self._init_layout()
        #self.observe(self.update)
        self.plot_ctrl.plot_text.value = True
        self.gc.calculate.observe(self.update)
        self.node_mapper.marker.marker_type.observe(self._on_marker_change)
        self.edge_mapper.marker.marker_type.observe(self._on_marker_change)
        self.plot_ctrl.plot_text.observe(self._on_marker_change)
        self.plot_ctrl.update_btn.observe(self._on_marker_change)
        self._handler = None
    
    def prepare_edge_mapper_data(self,gc):
        et = gc.edge.copy()
        cols = et.items.values
        fi = pd.DataFrame(et[cols[0]].unstack().copy())
        fi.columns = [cols[0]]

        for it in cols[1:]:
            new = pd.DataFrame(et[it].unstack().copy())
            new.columns = [it]
            fi = fi.combine_first(new)
        fi.index.levels[0].name = 'major'
        fi.index.levels[1].name = 'minor'
        dft = fi.reset_index()
        
        ti = list(zip(dft.minor.values,dft.major.values))
        dft.index = ti
        dft = dft.drop(['major','minor'],axis=1)
        return dft.ix[gc.G.edges()].copy()
        
    def _init_layout(self):
        params = ['size', 'line_width', 'fill_alpha', 'line_alpha', 'line_color', 'fill_color']
        for target in ['node','edge']:
            mapper = getattr(self,target+'_mapper')
            getattr(mapper).data_slicer.index_slicer.visible = False
            
    def update_source_df(self):
        """Datasources for plots managing"""
        def index_to_columns(df):
            old_index = df.index
            new_index = pd.MultiIndex.from_tuples(df.index)
            df.index = new_index
            df = df.reset_index()
            df.index = old_index
            columns = df.columns.values
            columns[:2] = ['major','minor']
            df.columns = columns
            return df
        def fix_pos_index(edge_pos,edge_source):
            changed = set(edge_source.index) - set(edge_pos.index)
            edge_source.index = pd.MultiIndex.from_tuples(edge_source.index)
            edge_pos.index = pd.MultiIndex.from_tuples(edge_pos.index)
            for e in changed:
                inv = (e[1],e[0])
                edge_pos.loc[inv,'minor'] = e[1]
                edge_pos.loc[inv,'major'] = e[0]
            return edge_pos
        self.node_source= self.node_mapper.output.join(self.node_tooltip.output)
        #self.edge_tooltip.output.index = pd.MultiIndex.from_tuples(self.edge_tooltip.output.index.values)
        #self.edge_mapper.output.index = pd.MultiIndex.from_tuples(self.edge_mapper.output.index.values)
        self.edge_source= self.edge_mapper.output.join(self.edge_tooltip.output) 
        
        node_pos = self.gc.layout.node['2d'].dropna(axis=1).copy()
        #edge_pos = self.gc.layout.edge['2d'].copy()
        
        self.output_node = pd.concat([self.node_source, node_pos], axis=1).fillna('NaN').copy()
        self.output_node.columns = [str(x) for x in self.output_node.columns]
        self.output_node.index = [str(x) for x in self.output_node.index]
        df = self.edge_source.copy()
        df['src'] = self.gc.edge['src'].values
        df['dst'] = self.gc.edge['dst'].values
        lay = self.gc.layout.edge['2d'].fillna(0).copy()
        #df.loc[:,'index'] = df['index'].map(str).values
        lay = lay.reset_index()
        lay['src'] = lay['index'].map(lambda x:x[0])
        lay['dst'] = lay['index'].map(lambda x:x[1])
        """for c in lay.columns:
            df[c] = np.nan
        rows = lay.iterrows()
        for ix,row in rows:
            cond_src = df['src']==ix[0]
            cond_dst = df['dst']==ix[1]
            df.loc[cond_src&cond_dst,lay.columns] = row.values"""
        self.output_edge = df.merge(lay,on=['src','dst'], left_index=True).fillna('NaN').copy()#edge_pos.merge(self.edge_source,on=['major','minor'],left_index=True)
        def _hack(x):
            try:
                return float(x)
            except:
                return str(x)
        self.output_edge = self.output_edge.applymap(_hack)
    
    def update_mappers(self):
        
        edge_mapper_data = self.gc.edge
        self.edge_mapper.data = edge_mapper_data
        self.node_mapper.data = self.gc.node
        self.node_tooltip.data = self.gc.node
        self.edge_tooltip.data = edge_mapper_data
    
    def update_tooltips(self):
        """Update tooltip data and inject it into the plot tooltips"""
        self.plot.tools[-1].tooltips = self.edge_tooltip.create_tooltip()
        self.plot.tools[-2].tooltips = self.node_tooltip.create_tooltip()

    def update(self, _=None):
        """Set up all the combined elements needed for the plot"""
        self.node_mapper.update()
        self.edge_mapper.update()
        self.update_mappers()
        self.update_source_df()
        self.update_tooltips()
        self.push_data()
        
    def push_data(self):
        """A function to push the content of the source DataFrame
        to a specific plot source
        """
        for col in self.output_node.columns:
            self.node_bokeh_source.data[col] = self.output_node[col].values
        
        for col in self.output_edge.columns:
            self.edge_bokeh_source.data[col] = self.output_edge[col].values
        push_notebook()
    
    def _on_marker_change(self, _=None):
        #clear_output()
        #self.init_plot()
        self.update()
        #return display(show(self.plot))
    
    def init_plot(self):
        """Handle plot init"""
        self.node_bokeh_source = bks.ColumnDataSource(self.output_node)
        self.edge_bokeh_source = bks.ColumnDataSource(self.output_edge)
        n_ttip = self.node_tooltip.create_tooltip()
        e_ttip = self.edge_tooltip.create_tooltip()

        self.plot = figure(width=self._width, height=self._height, webgl=False,
                           tools="pan,wheel_zoom,box_zoom,reset,resize,crosshair",)

        #node_marker = 'circle'#self.node_marker_free_sel.marker.value
        #edge_marker = self.edge_marker_free_sel.marker.value
        self.plot.title.text  ="Graph plot"
        self.plot.segment('x0', 'y0',
                          'x1', 'y1',
                          color='line_color',
                          line_width='line_width',
                          alpha='line_alpha',
                          source=self.edge_bokeh_source)

        edg_center = self.plot.scatter(source=self.edge_bokeh_source,
                                       x='cx', y='cy',
                                       line_color='line_color',
                                       fill_color='fill_color',
                                       alpha='fill_alpha',
                                       size='size',
                                       line_width='line_width',
                                       marker=self.edge_mapper.marker.marker_type.value
                                       )#,
                                       #marker=edge_marker)
        nod = self.plot.scatter(source=self.node_bokeh_source,
                                x='x', y='y',
                                line_color='line_color',
                                fill_color='fill_color',
                                fill_alpha='fill_alpha',
                                size='size',
                                line_width='line_width',#,
                                marker=self.node_mapper.marker.marker_type.value)
    
        if self.plot_ctrl.plot_text.value:
            self.plot.text(source=self.node_bokeh_source,
                           x='x', y='y',
                           text='label',
                           text_align='center')

        self.plot.add_tools(HoverTool(tooltips=n_ttip, renderers=[nod]))
        self.plot.add_tools(HoverTool(tooltips=e_ttip, renderers=[edg_center]))
        self.plot.background_fill_color="#eeeeee"
        self.plot.title_location="above"
        self.plot.title.text = "Xarxa GEANT"
        self.plot.title.text_font_size="20pt"
        self._handler = show(self.plot, notebook_handle=True)
        
    @property
    def snapshot(self, name='bokeh_scatter'):
        html =  notebook_div(self.plot)
        widget = shaoscript('html$N='+name)
        widget.value = html
        return widget

    def load_data(self):
        self.node_mapper.update()
        self.edge_mapper.update()
        self.update_mappers()
        
        self.update_source_df()
        self.init_plot()
        """ self.node_mapper.observe(self.update)
        self.edge_mapper.observe(self.update)
        self.node_tooltip.observe(self.update)
        self.edge_tooltip.observe(self.update)
        self.gc.calculate.observe(self.update)"""
        #self.update_tooltips()
       
    
    def show(self):
        return display(self.widget, show(self.plot))

ls_nombres=['NULL','IS','FI','NO','SE','EE','LV','LT','DK','IE','RU','UK','NL','PL','BY','BE','LU','FR','DE','UA',
             'CZ','CH','SK','AT','ES','PT','IT','SI','HR','HU','MD','RS','ME','RO','BG','TR','MK','GR','MT','CY','IL']
G=nx.DiGraph()
for i in range(1,41):
   # G.add_node(i, name=ls_nombres[i])
    G.add_node(i)
    
labels={}
for i in range(1,41):
    name=ls_nombres[i]
    labels[i]=name  

G.add_edge(1,8,BW=10,weight=10)
G.add_edge(1,11,BW=2.5,weight=2.5)
G.add_edge(2,4,BW=15,weight=15)
G.add_edge(2,4,BW=15,weight=15)
G.add_edge(3,4,BW=15,weight=15)
G.add_edge(3,8,BW=15,weight=15)
G.add_edge(4,8,BW=15,weight=15)
G.add_edge(8,18,BW=15,weight=15)
G.add_edge(5,8,BW=10,weight=10)
G.add_edge(5,6,BW=10,weight=10)
G.add_edge(6,7,BW=10,weight=10)
G.add_edge(7,13,BW=10,weight=10)
G.add_edge(8,12,BW=15,weight=15)
G.add_edge(9,11,BW=15,weight=15)
G.add_edge(9,11,BW=10,weight=10)
G.add_edge(10,8,BW=10,weight=10)
G.add_edge(10,18,BW=10,weight=10)
G.add_edge(11,40,BW=2.5,weight=2.5)
G.add_edge(11,15,BW=15,weight=15)
G.add_edge(11,17,BW=15,weight=15)
G.add_edge(11,25,BW=10,weight=10)
G.add_edge(11,39,BW=1,weight=1)
G.add_edge(12,15,BW=15,weight=15)
G.add_edge(12,18,BW=15,weight=15)
G.add_edge(13,18,BW=10,weight=10)
G.add_edge(13,20,BW=10,weight=10)
G.add_edge(13,14,BW=1,weight=1)
G.add_edge(13,19,BW=1,weight=1)
G.add_edge(16,18,BW=10,weight=10)
G.add_edge(16,17,BW=10,weight=10)
G.add_edge(17,21,BW=15,weight=15)
G.add_edge(17,24,BW=10,weight=10)
G.add_edge(18,20,BW=15,weight=15)
G.add_edge(18,21,BW=15,weight=15)
G.add_edge(18,23,BW=10,weight=10)
G.add_edge(18,39,BW=1,weight=1)
G.add_edge(18,40,BW=2.5,weight=2.5)
G.add_edge(20,22,BW=15,weight=15)
G.add_edge(21,26,BW=15,weight=15)
G.add_edge(21,24,BW=15,weight=15)
G.add_edge(22,23,BW=15,weight=15)
G.add_edge(22,29,BW=15,weight=15)
G.add_edge(23,27,BW=15,weight=15)
G.add_edge(23,37,BW=10,weight=10)
G.add_edge(24,26,BW=10,weight=10)
G.add_edge(24,25,BW=10,weight=10)
G.add_edge(26,38,BW=1,weight=1)
G.add_edge(27,28,BW=15,weight=15)
G.add_edge(28,29,BW=15,weight=15)
G.add_edge(28,32,BW=0.155,weight=0.155)
G.add_edge(30,33,BW=1,weight=1)
G.add_edge(29,31,BW=10,weight=10)
G.add_edge(29,34,BW=10,weight=10)
G.add_edge(29,33,BW=10,weight=10)
G.add_edge(33,35,BW=10,weight=10)
G.add_edge(33,34,BW=10,weight=10)
G.add_edge(34,37,BW=10,weight=10)
G.add_edge(34,36,BW=0.155,weight=0.155)
G.add_edge(34,35,BW=10,weight=10)
G=nx.relabel_nodes(G,labels)
G.node