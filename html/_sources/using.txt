=============
Notes on LFPy
=============


Units
=====

Units follow the NEURON conventions.
The units in LFPy for given quantities are:

+-------------+-----------+
| What        | Unit      |
+=============+===========+
| Potentials  | [mV]      |
+-------------+-----------+
| Currents    | [nA]      |
+-------------+-----------+
| conductance | [S/cm2]   |
+-------------+-----------+
| capacitance | [muF/cm2] |
+-------------+-----------+
| Coordinates | [mum]     |
+-------------+-----------+

Note: resistance, conductance and capacitance are usually specific values, i.e per membrane area (lowercase r_m, g, c_m)
Depending on the mechanism files, some may use different units altogether, but this should be taken care of internally by NEURON.


.. Documentation
.. ===============
.. 
.. To rebuild this documentation from the LFPy-release root folder, issue in terminal
.. ::
..     export LC_ALL=en_US.UTF-8
..     sphinx-build-2.* -b html documentation/sphinx_files/. html
