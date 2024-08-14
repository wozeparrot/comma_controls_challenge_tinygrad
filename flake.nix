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
              ];
            python = pkgs.python312;
          in [
            (python.withPackages python-packages)
          ];
        };
      }
    );
}