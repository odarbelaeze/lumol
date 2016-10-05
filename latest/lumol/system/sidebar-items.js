initSidebarItems({"constant":[["CONNECT_12","The particles are separated by one bond"],["CONNECT_13","The particles are separated by two bonds"],["CONNECT_14","The particles are separated by three bonds"],["CONNECT_FAR","The particles are separated by more than three bonds"]],"enum":[["CellShape","The shape of a cell determine how we will be able to compute the periodic boundaries condition."]],"fn":[["moltype","Get the molecule type of the given `molecule` containing the `particles`. This type can be used to identify all the molecules containing the same bonds and particles (see `System::moltype` for more information)."]],"mod":[["compute","Computing properties of a system"],["velocities","This module provides some ways to initialize the velocities in a `System`"]],"struct":[["Angle","An `Angle` formed by the atoms at indexes `i`, `j` and `k`"],["Bond","A `Bond` between the atoms at indexes `i` and `j`"],["Connectivity","The `Connectivity` bitflag encode the topological distance between two particles in the molecule, i.e. the number of bonds between the particles."],["Dihedral","A `Dihedral` angle formed by the atoms at indexes `i`, `j`, `k` and `m`"],["ElementData","Data about one \"extended\" chemical element."],["EnergyCache","This is a cache for energy computation."],["EnergyEvaluator","An helper struct to evaluate energy components of a system."],["Molecule","A molecule is the basic building block for a topology. It contains data about the connectivity (bonds, angles, dihedrals) in the system."],["Particle","The Particle type hold basic data about a particle in the system. It is self contained, so that it will be easy to send data between parrallels processes."],["ParticleKind","A particle kind. Particles with the same name will have the same kind. This is used for faster potential lookup."],["PeriodicTable","The `PeriodicTable` struct give access to elements information, and give a way to register new elements in the list."],["System","The System type hold all the data about a simulated system."],["UnitCell","An UnitCell defines the system physical boundaries."]],"type":[["Permutations","Particles permutations:. Indexes are given in the `(old, new)` form."]]});