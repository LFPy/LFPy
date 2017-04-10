#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Copyright (C) 2012 Computational Neuroscience Group, NMBU.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

"""

def _test(verbosity=1):
    """run all unit tests included with LFPy.
    
    Parameters
    ----------
    verbosity : int
        unittest.TestCase verbosity level, default is 1
    
    Examples
    --------
    From Python:
    >>> import LFPy
    >>> LFPy.run_tests()
    
    Using external testing framework (nose, py.test etc.) from command line
    $ cd <path to LFPy>
    $ nosetests-2.7
    
    Run single test modules
    $ cd <path to LFPy>
    $ nosetests-2.7 LFPy/test/test_cell.py
    
    """
    # import methods here to avoid polluting LFPy.test namespace
    from .test_cell import testCell
    from .test_eegmegcalc import testMEG, testFourSphereVolumeConductor, testInfiniteVolumeConductor
    from .test_alias_method import testAliasMethod
    from .test_recextelectrode import testRecExtElectrode
    from .test_lfpcalc import testLfpCalc
    from .test_misc import testMisc
    from .test_pointprocess import testPointProcess, testSynapse, testStimIntElectrode 
    from .test_inputgenerators import testInputGenerators
    from .test_templatecell import testTemplateCell
    from .test_networkcell import testNetworkCell
    from .test_network import testNetworkPopulation, testNetwork
    import unittest

    print('\ntest LFPy.Cell class and methods:')
    suite = unittest.TestLoader().loadTestsFromTestCase(testCell)
    unittest.TextTestRunner(verbosity=verbosity).run(suite)

    print('\ntest LFPy.TemplateCell class and methods:')
    suite = unittest.TestLoader().loadTestsFromTestCase(testTemplateCell)
    unittest.TextTestRunner(verbosity=verbosity).run(suite)

    print('\ntest LFPy.lfpcalc methods:')
    suite = unittest.TestLoader().loadTestsFromTestCase(testLfpCalc)
    unittest.TextTestRunner(verbosity=verbosity).run(suite)

    print('\ntest LFPy.RecExtElectrode class and methods:')
    suite = unittest.TestLoader().loadTestsFromTestCase(testRecExtElectrode)
    unittest.TextTestRunner(verbosity=verbosity).run(suite)

    print('\ntest LFPy.NetworkCell class and methods:')
    suite = unittest.TestLoader().loadTestsFromTestCase(testNetworkCell)
    unittest.TextTestRunner(verbosity=verbosity).run(suite)

    print('\ntest LFPy.NetworkPopulation class and methods:')
    suite = unittest.TestLoader().loadTestsFromTestCase(testNetworkPopulation)
    unittest.TextTestRunner(verbosity=verbosity).run(suite)

    print('\ntest LFPy.Network class and methods:')
    suite = unittest.TestLoader().loadTestsFromTestCase(testNetwork)
    unittest.TextTestRunner(verbosity=verbosity).run(suite)

    print('\ntest LFPy.MEG class and methods:')
    suite = unittest.TestLoader().loadTestsFromTestCase(testMEG)
    unittest.TextTestRunner(verbosity=verbosity).run(suite)

    print('\ntest LFPy.FourSphereVolumeConductor class and methods:')
    suite = unittest.TestLoader().loadTestsFromTestCase(testFourSphereVolumeConductor)
    unittest.TextTestRunner(verbosity=verbosity).run(suite)

    print('\ntest LFPy.InfiniteVolumeConductor class and methods:')
    suite = unittest.TestLoader().loadTestsFromTestCase(testInfiniteVolumeConductor)
    unittest.TextTestRunner(verbosity=verbosity).run(suite)

    print('\ntest LFPy.alias_method methods:')
    suite = unittest.TestLoader().loadTestsFromTestCase(testAliasMethod)
    unittest.TextTestRunner(verbosity=verbosity).run(suite)

    print('\ntest LFPy.PointProcess class and methods:')
    suite = unittest.TestLoader().loadTestsFromTestCase(testPointProcess)
    unittest.TextTestRunner(verbosity=verbosity).run(suite)

    print('\ntest LFPy.Synapse class and methods:')
    suite = unittest.TestLoader().loadTestsFromTestCase(testSynapse)
    unittest.TextTestRunner(verbosity=verbosity).run(suite)

    print('\ntest LFPy.StimIntElectrode class and methods:')
    suite = unittest.TestLoader().loadTestsFromTestCase(testStimIntElectrode)
    unittest.TextTestRunner(verbosity=verbosity).run(suite)

    print('\ntest LFPy.inputgenerators methods:')
    suite = unittest.TestLoader().loadTestsFromTestCase(testInputGenerators)
    unittest.TextTestRunner(verbosity=verbosity).run(suite)

    print('\ntest misc. methods:')
    suite = unittest.TestLoader().loadTestsFromTestCase(testMisc)
    unittest.TextTestRunner(verbosity=verbosity).run(suite)