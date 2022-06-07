import plotly.graph_objects as go
import plotly
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from persim import plot_diagrams
import matplotlib as mpl
inline_rc = dict(mpl.rcParams)

### plot


def plot_3d(X, color = None, size = 2, opacity = 1, width = 300, height=200, margins=[0,0,0,0], title="", colorscale="viridis", transparent=False) :
    if colorscale == "viridis":
        cs = plotly.colors.sequential.Viridis
    else :
        cs = plotly.colors.diverging.RdGy

    if transparent:
        layout = go.Layout( scene=dict( aspectmode='data'), width=width, height=height, margin=go.layout.Margin( l=margins[0], r=margins[1], b=margins[2], t=margins[3]), template="none", title=title,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
        )
    else :
        layout = go.Layout( scene=dict( aspectmode='data'), width=width, height=height, margin=go.layout.Margin( l=margins[0], r=margins[1], b=margins[2], t=margins[3]), template="none", title=title,
        )

    if color is None:
        data_plot = go.Scatter3d(x = X[:,0], y = X[:,1], z = X[:,2], mode='markers', marker=dict(size=size, opacity=opacity))
    else :
        data_plot = go.Scatter3d(x = X[:,0], y = X[:,1], z = X[:,2], mode='markers', marker=dict(size=size,color=color, colorscale=cs, opacity=opacity))

    return go.Figure([data_plot], layout=layout)


def plot_2d(X, color = None, size = 2, opacity = 1, width = 300, height=200, margins=[0,0,0,0], title="", colorscale="viridis", marker_symbol="circle") :
    if colorscale == "viridis":
        cs = plotly.colors.sequential.Viridis
    else :
        cs = plotly.colors.diverging.RdGy

    layout = go.Layout( scene=dict( aspectmode='data'), width=width, height=height, margin=go.layout.Margin( l=margins[0], r=margins[1], b=margins[2], t=margins[3]), template="simple_white", title=title)

    if color is None:
        data_plot = go.Scatter(x = X[:,0], y = X[:,1], mode='markers', marker=dict(size=size,opacity=opacity, symbol=marker_symbol))
    else :
        data_plot = go.Scatter(x = X[:,0], y = X[:,1], mode='markers', marker=dict(size=size,color=color, colorscale=cs,opacity=opacity, symbol=marker_symbol))

    return go.Figure([data_plot], layout=layout)


def plot_2d_(X, color = None, point_size = 10, width=4, height=4):
    fig = plt.figure(figsize=(width,height))
    if color is None:
        plt.scatter(X[:,0], X[:,1], s=point_size)
    else :
        plt.scatter(X[:,0], X[:,1], c=color, s=point_size)
    return fig




def plot_with_images_on_top(X, n_subsample, labels, images, ax, seed=0):
    indices = np.array(range(X.shape[0]))
    #indices = np.where(digits['target']==3)[0]

    dm_X = distance_matrix(X[indices],X[indices])
    gp, _ = getGreedyPerm(dm_X)
    subsample = np.array(gp[:n_subsample])

    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='Spectral', s=5)

    for x0, y0, im in zip(X[indices[subsample],0], X[indices[subsample],1],images[indices[subsample]]):
        ab = AnnotationBbox(OffsetImage(im, cmap='binary',zoom=2), (x0, y0))
        ax.add_artist(ab)



def plot_weighted_graph_3d(vertices, edges, weights_vertices, weights_edges, vertex_size = 10, line_size=2, alpha_lines = 0.5, layout=None):
    x = vertices[:,0]
    y = vertices[:,1]
    z = vertices[:,2]

    x_lines = list()
    y_lines = list()
    z_lines = list()

    for edge in edges:
        for i in range(2):
            x_lines.append(x[edge[i]])
            y_lines.append(y[edge[i]])
            z_lines.append(z[edge[i]])
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)

    drawings = [go.Scatter3d(x = vertices[:,0], y = vertices[:,1], z = vertices[:,2], mode='markers', marker=dict(size=vertex_size), opacity=1)]
    trace2 = go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode='lines',
        name='lines',
        line=dict(color=weights_edges, width=line_size)
    )

    drawings.append(trace2)

    if layout == None:
        fig = go.Figure(drawings)
    else :
        fig = go.Figure(drawings, layout = layout)
    fig.show()


def save_PD(dgms, location, title, lims, inline_rc,labels=["","$H^1$","$H^2$"], plot_only=[1,2], figsize=(2,2), size=20):
    plt.figure(figsize=figsize)
    #plt.gca().set_aspect('equal')
    if len(lims) == 2:
        lims = lims + lims
    plot_diagrams(dgms,plot_only=plot_only,title=title,size=size,labels=labels,show=False,xy_range=[lims[0],lims[1],lims[2],lims[3]])
    plt.xlabel("")
    plt.ylabel("")
    plt.xlim(lims[:2])
    plt.ylim(lims[2:])
    plt.xticks([])
    plt.yticks([])
    #plt.axis("off")
    #plt.tight_layout()
    plt.savefig(location, bbox_inches = 'tight', transparent=True)  
    mpl.rcParams.update(inline_rc)
    plt.show()


def plot_sw(sol,basis,dth,dgms, lims, location, title, size = 5, save=True):
    n_gens1 = len(basis)
    appear1 = np.nonzero(sol[:n_gens1])

    fig = plt.figure(figsize=(size,size))
    plt.gca().set_aspect('equal')

    ax = plt.gca()
    rect1 = patches.Rectangle((0, 0), dth, 1000, color="grey", alpha = 0.2)
    ax.add_patch(rect1)
    triang = patches.Polygon([[0,0],[dth,0],[dth,dth]], color="white")
    ax.add_patch(triang)

    plt.rcParams.update({'font.size' : 13})
    plot_diagrams(dgms, title=title, plot_only=[1], labels=["$H^0$","$H^1$","$H^2$"], show=False, size = 10, xy_range=[lims[0],lims[1],lims[0],lims[1]])

    mpl.rcParams.update(inline_rc)
    ax.scatter(dgms[1][appear1, 0], dgms[1][appear1, 1], 65, 'r', 'o', label="$sw_1$")
    ax.scatter(dgms[1][appear1, 0], dgms[1][appear1, 1], 35, 'w', 'o')
    ax.scatter(dgms[1][appear1, 0], dgms[1][appear1, 1], 10, 'blue', 'o')
    ax.scatter([],[], 65, 'grey', 's', alpha=0.2, label="span")

    plt.xlabel("")
    plt.ylabel("")
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xticks([])
    plt.yticks([])

    ax.legend()
    ax.set_xlabel("")
    ax.set_ylabel("")

    if save :
        plt.savefig(location, bbox_inches = 'tight')  
