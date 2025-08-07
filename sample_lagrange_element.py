#!/usr/bin/env python3
"""
Generate a sample Lagrange element file to check on vertex ordering. Change
the code on the indicated lines for desired element type and order or specify
them as arguments.

Written by Jens Ulrich Kreber <ju.kreber@gmail.com> for the Chair for
Electromagnetic Theory at Saarland University
(https://www.uni-saarland.de/lehrstuhl/dyczij-edlinger.html)
"""

# %%
import numpy as np
import vtk
import vtk.util.numpy_support as vnp  # type: ignore
import matplotlib.pyplot as plt
from node_ordering import node_ordering

# %%
VTK_CELLTYPES = {
    "triangle": vtk.VTK_LAGRANGE_TRIANGLE,
    "tetrahedron": vtk.VTK_LAGRANGE_TETRAHEDRON,
    "quadrilateral": vtk.VTK_LAGRANGE_QUADRILATERAL,
    "hexahedron": vtk.VTK_LAGRANGE_HEXAHEDRON,
    "wedge": vtk.VTK_LAGRANGE_WEDGE,
}

# %%
# Change to desired element here
element_type = "hexahedron"
element_order = 2

assert (element_type) in VTK_CELLTYPES.keys()
assert element_order in range(1, 11), "Element order must be between 1 and 10"

# %%
# Create points for the element
points = node_ordering(element_type, element_order)

# %%
points_vtk_data = vnp.numpy_to_vtk(points)
points_vtk = vtk.vtkPoints()
points_vtk.SetData(points_vtk_data)

num_points = points.shape[0]
cell_data = np.concatenate(
    (np.array([num_points]), np.arange(num_points)), axis=0
)
id_type_np = vnp.ID_TYPE_CODE  # the numpy dtype for vtkIdTypeArray
cell_data_vtk = vnp.numpy_to_vtk(cell_data, array_type=vtk.VTK_ID_TYPE)
cells = vtk.vtkCellArray()
cells.SetCells(1, cell_data_vtk)

# Simply use point index as data (can be visualized as a field in ParaView)
pointdata = np.arange(num_points, dtype=np.float64)
pointdata_vtk = vnp.numpy_to_vtk(pointdata)
pointdata_vtk.SetName("point_numbers")

ugrid = vtk.vtkUnstructuredGrid()
ugrid.SetPoints(points_vtk)
ugrid.SetCells(VTK_CELLTYPES[element_type], cells)
pointdata_container = ugrid.GetPointData()
pointdata_container.SetScalars(pointdata_vtk)

writer = vtk.vtkXMLUnstructuredGridWriter()
writer.SetInputDataObject(ugrid)
writer.SetDataModeToAscii()
writer.SetCompressorTypeToNone()
writer.SetFileName("lagrange_sample.vtu")
writer.Write()
# Now let's create the tensorized points, first 1d
tensorized_points = np.linspace(0, 1, element_order + 1)


tensorized_points_3d = np.array(
    np.meshgrid(
        tensorized_points,
        tensorized_points,
        tensorized_points,
        indexing="ij",
    )
).T.reshape(-1, 3)


# %%
# Create 3D scatter plot of node ordering
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Create scatter plot for Lagrange points
scatter_lagrange = ax.scatter(
    points[:, 0],
    points[:, 1],
    points[:, 2],
    c=pointdata,
    cmap="rainbow",
    s=100,
    alpha=1.0,
    label="Lagrange Points",
)

# Create scatter plot for tensorized points
tensorized_pointdata = np.arange(len(tensorized_points_3d), dtype=np.float64)
# scatter_tensor = ax.scatter(
#     tensorized_points_3d[:, 0],
#     tensorized_points_3d[:, 1],
#     tensorized_points_3d[:, 2],
#     c=tensorized_pointdata,
#     cmap="viridis",
#     s=50,
#     alpha=0.6,
#     marker="^",
#     label="Tensorized Points",
# )

# Add node numbers as text labels for Lagrange points
for i, (x, y, z) in enumerate(points):
    ax.text(x, y, z - 0.05, f"  VTK{i}", fontsize=8, color="blue")

# Add node numbers as text labels for tensorized points
for i, (x, y, z) in enumerate(tensorized_points_3d):
    ax.text(x, y, z + 0.05, f"  T{i}", fontsize=7, color="green")

# Set labels and title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title(
    f"{element_type.capitalize()} Element (Order "
    f"{element_order}) - Node Ordering Comparison"
)

# Add colorbar for Lagrange points
plt.colorbar(
    scatter_lagrange, ax=ax, shrink=0.5, aspect=5, label="Node index in VTK"
)


# Look down in positive direction for x/y.
ax.view_init(elev=40, azim=-115)

# Show the plot
plt.tight_layout()
plt.show()

# %%
