{ fetchurl, lib, stdenvNoCC }:

stdenvNoCC.mkDerivation rec {
  pname = "nvidia-mathdx";
  version = "26.03.0-cuda13";

  src = fetchurl {
    url = "https://developer.nvidia.com/downloads/compute/cuSOLVERDx/redist/cuSOLVERDx/cuda13/nvidia-mathdx-26.03.0-cuda13.tar.gz";
    hash = "sha256-lcJ0QFEC9qJj956wREe9dTn2bOZJd5vY4iX8uMxxWU0=";
  };

  sourceRoot = ".";
  dontBuild = true;

  installPhase = ''
    runHook preInstall

    mathdx_include="$(find . -path '*/include/cusolverdx.hpp' -print -quit)"
    if [ -z "$mathdx_include" ]; then
      echo "cusolverdx.hpp not found in MathDx archive" >&2
      exit 1
    fi

    mathdx_root="$(dirname "$(dirname "$mathdx_include")")"
    mkdir -p "$out"
    cp -R "$mathdx_root"/. "$out"/

    runHook postInstall
  '';

  meta = with lib; {
    description = "NVIDIA MathDx SDK containing cuSolverDx";
    homepage = "https://developer.nvidia.com/cusolverdx-downloads";
    license = licenses.unfreeRedistributable or licenses.unfree;
    platforms = platforms.linux;
  };
}
