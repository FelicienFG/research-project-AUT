from setuptools import setup, Extension

from setuptools.command.build_py import build_py as _build_py

class build_py(_build_py):
    def run(self):
        self.run_command("build_ext")
        return super().run()

module_makespan_solver = Extension('_makespan_solver', sources=['bindings/makespan_solver.i', './cpp/makespan_solver.cpp']
    , libraries=['m','z'] 
    , swig_opts = ["-c++", "-Wall"],
    include_dirs=['bindings'])


setup(name='makespan_solver_package', version='1.0',
       py_modules = ['makespan_solver'],
       cmdclass = {'build_py' : build_py},       
       options = {"build_ext": {"inplace": False}},
 description="This package contains a module for computing the makespan of a DAG task, given its subtask's priority list and wcet list",
  ext_modules=[module_makespan_solver],
    package_dir = {'' : 'bindings'})