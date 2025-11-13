import os


def set_vc_envs():
    os.environ["PATH"] += r";c:\_devel\vs2022ent\VC\Tools\MSVC\14.37.32822\bin\Hostx64\x64"
    os.environ[
        "INCLUDE"] = r"C:\_devel\vs2022ent\VC\Tools\MSVC\14.37.32822\include;C:\_devel\vs2022ent\VC\Tools\MSVC\14.37.32822\ATLMFC\include;C:\_devel\vs2022ent\VC\Auxiliary\VS\include;C:\Program Files (x86)\Windows Kits\10\include\10.0.22621.0\ucrt;C:\Program Files (x86)\Windows Kits\10\include\10.0.22621.0\um;C:\Program Files (x86)\Windows Kits\10\include\10.0.22621.0\shared;C:\Program Files (x86)\Windows Kits\10\include\10.0.22621.0\winrt;C:\Program Files (x86)\Windows Kits\10\include\10.0.22621.0\cppwinrt"
    os.environ[
        "LIB"] = r"C:\_devel\vs2022ent\VC\Tools\MSVC\14.37.32822\ATLMFC\lib\x64;C:\_devel\vs2022ent\VC\Tools\MSVC\14.37.32822\lib\x64;C:\Program Files (x86)\Windows Kits\NETFXSDK\4.8\lib\um\x64;C:\Program Files (x86)\Windows Kits\10\lib\10.0.22621.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\lib\10.0.22621.0\um\x64"
    os.environ["DISTUTILS_USE_SDK"] = "1"
    os.environ["MSSdk"] = "1"
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'

    # Настройка кэширования
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = r'c:\_devel\_triton_cache\inductor'
    os.environ['TRITON_CACHE_DIR'] = r'c:\_devel\_triton_cache\compile'
