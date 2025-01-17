{
  description = "";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    tinygrad.url = "github:wozeparrot/tinygrad-nix";
  };

  outputs = inputs @ {
    nixpkgs,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem
    (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            inputs.tinygrad.overlays.default
          ];
        };
      in {
        devShell = pkgs.mkShell {
          packages = let
            python-packages = p:
              with p; [
                (tinygrad.override {
                  rocmSupport = true;
                })
                numpy
                onnx
                onnxruntime
                matplotlib
                seaborn
                tqdm
                wandb
              ];
            python = pkgs.python311;
          in [
            (python.withPackages python-packages)
            pkgs.clang
            pkgs.inotify-tools
          ];
        };
      }
    );
}
