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


def _test(verbosity=2):
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

    # list of test cases
    suites = []
    suites += [unittest.TestLoader().loadTestsFromTestCase(testCell)]
    suites += [unittest.TestLoader().loadTestsFromTestCase(testTemplateCell)]
    suites += [unittest.TestLoader().loadTestsFromTestCase(testLfpCalc)]
    suites += [unittest.TestLoader().loadTestsFromTestCase(testRecExtElectrode)]
    suites += [unittest.TestLoader().loadTestsFromTestCase(testNetworkCell)]
    suites += [unittest.TestLoader().loadTestsFromTestCase(testNetworkPopulation)]
    suites += [unittest.TestLoader().loadTestsFromTestCase(testNetwork)]
    suites += [unittest.TestLoader().loadTestsFromTestCase(testMEG)]
    suites += [unittest.TestLoader().loadTestsFromTestCase(testFourSphereVolumeConductor)]
    suites += [unittest.TestLoader().loadTestsFromTestCase(testInfiniteVolumeConductor)]
    suites += [unittest.TestLoader().loadTestsFromTestCase(testAliasMethod)]
    suites += [unittest.TestLoader().loadTestsFromTestCase(testPointProcess)]
    suites += [unittest.TestLoader().loadTestsFromTestCase(testSynapse)]
    suites += [unittest.TestLoader().loadTestsFromTestCase(testStimIntElectrode)]
    suites += [unittest.TestLoader().loadTestsFromTestCase(testInputGenerators)]
    suites += [unittest.TestLoader().loadTestsFromTestCase(testMisc)]

    unittest.TextTestRunner(verbosity=verbosity).run(unittest.TestSuite(suites))
