// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Also licensed under MIT license, at your choice.

//! Load rendering shaders.

mod preprocess;

use std::{collections::HashSet, fs, path::Path};

use wgpu::Device;

use crate::engine::{BindType, Engine, Error, ShaderId};

pub const PATHTAG_REDUCE_WG: u32 = 256;
pub const PATH_BBOX_WG: u32 = 256;
pub const PATH_COARSE_WG: u32 = 256;
pub const PATH_DRAWOBJ_WG: u32 = 256;

pub struct Shaders {
    pub pathtag_reduce: ShaderId,
    pub pathtag_scan: ShaderId,
    pub path_coarse: ShaderId,
    pub backdrop: ShaderId,
    pub fine: ShaderId,
}

// Shaders for the full pipeline
pub struct FullShaders {
    pub pathtag_reduce: ShaderId,
    pub pathtag_scan: ShaderId,
    pub bbox_clear: ShaderId,
    pub pathseg: ShaderId,
    pub draw_reduce: ShaderId,
    pub draw_leaf: ShaderId,
    pub binning: ShaderId,
    pub tile_alloc: ShaderId,
    pub path_coarse: ShaderId,
    pub backdrop: ShaderId,
    pub coarse: ShaderId,
    pub fine: ShaderId,
}

fn load_shader_sources() -> (String, String, String, String, String, String, String, String, String, String, String, String) {
    // When targeting WASM, we need to directly bundle shader source strings at compile-time since
    // we cannot read from the local build directory at runtime.
    let mut src = "".to_string();
    #[cfg(not(target_arch = "wasm32"))]
    {
        let shader_dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/shader"));
        let read_shader =
            |path: &str| fs::read_to_string(shader_dir.join(path.to_string() + ".wgsl")).unwrap();
        return (
            read_shader("pathtag_reduce"),
            read_shader("pathtag_scan"),
            read_shader("bbox_clear"),
            read_shader("pathseg"),
            read_shader("draw_reduce"),
            read_shader("draw_leaf"),
            read_shader("binning"),
            read_shader("tile_alloc"),
            read_shader("path_coarse"),
            read_shader("backdrop"),
            read_shader("coarse"),
            read_shader("fine"),
        );
    }
    #[cfg(target_arch = "wasm32")]
    {
        return (
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/shader", "/pathtag_reduce.wgsl")).to_string(),
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/shader", "/pathtag_scan.wgsl")).to_string(),
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/shader", "/bbox_clear.wgsl")).to_string(),
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/shader", "/pathseg.wgsl")).to_string(),
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/shader", "/draw_reduce.wgsl")).to_string(),
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/shader", "/draw_leaf.wgsl")).to_string(),
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/shader", "/binning.wgsl")).to_string(),
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/shader", "/tile_alloc.wgsl")).to_string(),
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/shader", "/path_coarse_full.wgsl")).to_string(),
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/shader", "/coarse.wgsl")).to_string(),
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/shader", "/backdrop_dyn.wgsl")).to_string(),
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/shader", "/fine.wgsl")).to_string(),
        );
    }
}

pub fn init_shaders(device: &Device, engine: &mut Engine) -> Result<Shaders, Error> {
    let shader_dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/shader"));
    let imports = preprocess::get_imports(shader_dir);
    let (
        pathtag_reduce_src,
        pathtag_scan_src,
        bbox_clear_src,
        pathseg_src,
        draw_reduce_src,
        draw_leaf_src,
        binning_src,
        tile_alloc_src,
        path_coarse_src,
        coarse_src,
        backdrop_src,
        fine_src,
    ) = load_shader_sources();

    let empty = HashSet::new();
    let pathtag_reduce = engine.add_shader(
        device,
        preprocess::preprocess(&pathtag_reduce_src, &empty, &imports).into(),
        &[
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::Buffer,
        ],
    )?;
    let pathtag_scan = engine.add_shader(
        device,
        preprocess::preprocess(&pathtag_scan_src, &empty, &imports).into(),
        &[
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::Buffer,
        ],
    )?;
    let path_coarse_config = HashSet::new();
    // path_coarse_config.add("cubics_out");

    let path_coarse = engine.add_shader(
        device,
        preprocess::preprocess(&path_coarse_src, &path_coarse_config, &imports).into(),
        &[
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::Buffer,
            BindType::Buffer,
        ],
    )?;
    let backdrop = engine.add_shader(
        device,
        preprocess::preprocess(&backdrop_src, &empty, &imports).into(),
        &[BindType::BufReadOnly, BindType::Buffer],
    )?;
    let fine = engine.add_shader(
        device,
        preprocess::preprocess(&fine_src, &empty, &imports).into(),
        &[
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::Buffer,
        ],
    )?;
    Ok(Shaders {
        pathtag_reduce,
        pathtag_scan,
        path_coarse,
        backdrop,
        fine,
    })
}

pub fn full_shaders(device: &Device, engine: &mut Engine) -> Result<FullShaders, Error> {
    let shader_dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/shader"));
    let imports = preprocess::get_imports(shader_dir);
    let (
        pathtag_reduce_src,
        pathtag_scan_src,
        bbox_clear_src,
        pathseg_src,
        draw_reduce_src,
        draw_leaf_src,
        binning_src,
        tile_alloc_src,
        path_coarse_full_src,
        coarse_src,
        backdrop_dyn_src,
        fine_src,
    ) = load_shader_sources();

    let empty = HashSet::new();
    let mut full_config = HashSet::new();
    full_config.insert("full".into());
    let pathtag_reduce = engine.add_shader(
        device,
        preprocess::preprocess(&pathtag_reduce_src, &full_config, &imports).into(),
        &[
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::Buffer,
        ],
    )?;
    let pathtag_scan = engine.add_shader(
        device,
        preprocess::preprocess(&pathtag_scan_src, &full_config, &imports).into(),
        &[
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::Buffer,
        ],
    )?;
    let bbox_clear = engine.add_shader(
        device,
        preprocess::preprocess(&bbox_clear_src, &empty, &imports).into(),
        &[BindType::BufReadOnly, BindType::Buffer],
    )?;
    let pathseg = engine.add_shader(
        device,
        preprocess::preprocess(&pathseg_src, &full_config, &imports).into(),
        &[
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::Buffer,
            BindType::Buffer,
        ],
    )?;
    let draw_reduce = engine.add_shader(
        device,
        preprocess::preprocess(&draw_reduce_src, &empty, &imports).into(),
        &[
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::Buffer,
        ],
    )?;
    let draw_leaf = engine.add_shader(
        device,
        preprocess::preprocess(&draw_leaf_src, &empty, &imports).into(),
        &[
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::Buffer,
            BindType::Buffer,
        ],
    )?;
    let binning = engine.add_shader(
        device,
        preprocess::preprocess(&binning_src, &empty, &imports).into(),
        &[
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::Buffer,
            BindType::Buffer,
            BindType::Buffer,
            BindType::Buffer,
        ],
    )?;
    let tile_alloc = engine.add_shader(
        device,
        preprocess::preprocess(&tile_alloc_src, &empty, &imports).into(),
        &[
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::Buffer,
            BindType::Buffer,
            BindType::Buffer,
        ],
    )?;

    let path_coarse = engine.add_shader(
        device,
        preprocess::preprocess(&path_coarse_full_src, &full_config, &imports).into(),
        &[
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::Buffer,
            BindType::Buffer,
            BindType::Buffer,
        ],
    )?;
    let backdrop = engine.add_shader(
        device,
        preprocess::preprocess(&backdrop_dyn_src, &empty, &imports).into(),
        &[
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::Buffer,
        ],
    )?;
    let coarse = engine.add_shader(
        device,
        preprocess::preprocess(&coarse_src, &empty, &imports).into(),
        &[
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::Buffer,
            BindType::Buffer,
        ],
    )?;
    let fine = engine.add_shader(
        device,
        preprocess::preprocess(&fine_src, &full_config, &imports).into(),
        &[
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::Buffer,
            BindType::BufReadOnly,
        ],
    )?;
    Ok(FullShaders {
        pathtag_reduce,
        pathtag_scan,
        bbox_clear,
        pathseg,
        draw_reduce,
        draw_leaf,
        binning,
        tile_alloc,
        path_coarse,
        backdrop,
        coarse,
        fine,
    })
}
