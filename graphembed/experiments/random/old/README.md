Scripts used to generate random graphs, before a refactoring of the way I'm
doing grid runs. They also contain highly experimental things, such as sampling
from the Riemannian measure, which turned out to be much harder on the SPD
manifold. The reason is that we do not have a coordinate system that allows us
to easily constraint the point within a geodesic ball. We do have such a
representation in elliptical and hyperbolic spaces via polar coordinates.

Big parts of the scripts from this directory are used in the scripts under
'experiments/random' (the parent directory).
