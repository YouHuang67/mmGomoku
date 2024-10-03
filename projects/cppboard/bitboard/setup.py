from distutils.core import setup, Extension

sources = ['board_wrap.cxx', 'board.cpp', 
           'board_bits.cpp', 'init.cpp', 
           'lineshapes.cpp', 'pns.cpp', 'shapes.cpp']

module = Extension(
    '_board', sources=sources, 
    extra_compile_args=['/O2'], 
    language='c++'
)

setup(name='board',
      ext_modules=[module],
      py_modules=['board'])