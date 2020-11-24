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
    from .test_eegmegcalc import testMEG, testFourSphereVolumeConductor, \
        testInfiniteVolumeConductor, testOneSphereVolumeConductor
    from .test_alias_method import testAliasMethod
    from .test_recextelectrode import testRecExtElectrode
    from .test_lfpcalc import testLfpCalc
    from .test_misc import testMisc
    from .test_pointprocess import testPointProcess, testSynapse, \
        testStimIntElectrode
    from .test_inputgenerators import testInputGenerators
    from .test_templatecell import testTemplateCell
    from .test_networkcell import testNetworkCell
    from .test_network import testNetworkPopulation, testNetwork
    from .test_tools import testTools
    from .test_imem import testImem
    from .test_morphology_import import testMorphologyImport
    from unittest import TestSuite, TestLoader, TextTestRunner

    # list of test cases
    suites = []
    suites += [TestLoader().loadTestsFromTestCase(testCell)]
    suites += [TestLoader().loadTestsFromTestCase(testTemplateCell)]
    suites += [TestLoader().loadTestsFromTestCase(testLfpCalc)]
    suites += [TestLoader().loadTestsFromTestCase(testRecExtElectrode)]
    suites += [TestLoader().loadTestsFromTestCase(testNetworkCell)]
    suites += [TestLoader().loadTestsFromTestCase(testNetworkPopulation)]
    suites += [TestLoader().loadTestsFromTestCase(testNetwork)]
    suites += [TestLoader().loadTestsFromTestCase(testMEG)]
    suites += [TestLoader().loadTestsFromTestCase(testFourSphereVolumeConductor)]
    suites += [TestLoader().loadTestsFromTestCase(testInfiniteVolumeConductor)]
    suites += [TestLoader().loadTestsFromTestCase(testOneSphereVolumeConductor)]
    suites += [TestLoader().loadTestsFromTestCase(testAliasMethod)]
    suites += [TestLoader().loadTestsFromTestCase(testPointProcess)]
    suites += [TestLoader().loadTestsFromTestCase(testSynapse)]
    suites += [TestLoader().loadTestsFromTestCase(testStimIntElectrode)]
    suites += [TestLoader().loadTestsFromTestCase(testInputGenerators)]
    suites += [TestLoader().loadTestsFromTestCase(testImem)]
    suites += [TestLoader().loadTestsFromTestCase(testMisc)]
    suites += [TestLoader().loadTestsFromTestCase(testTools)]
    suites += [TestLoader().loadTestsFromTestCase(testMorphologyImport)]
    TextTestRunner(verbosity=verbosity).run(TestSuite(suites))
