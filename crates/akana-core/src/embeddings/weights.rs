//! TurboQuant 2-bit dequantization engine and minimal NPZ parser.
//!
//! Parses the `turboquant_weights.npz` file format (ZIP archive of NumPy arrays)
//! and dequantizes 2-bit packed indices into f32 embedding vectors.
//!
//! ## Format
//!
//! The NPZ file contains:
//! - `packed_indices`: `(vocab_size, dim/4)` u8 array — each byte packs 4 × 2-bit values
//! - `scales`: `(vocab_size,)` f16 array — per-token scaling factors
//! - `bits`: scalar i64 = 2
//! - `dim`: scalar i64 = 256
//! - `use_qjl`: scalar bool = false

use std::io::{self, Cursor, Read};

/// Embedding dimension for this model.
pub const EMBEDDING_DIM: usize = 256;

/// Number of 2-bit values packed into each u8.
const PACK_FACTOR: usize = 4;

/// Number of bytes per packed row: dim / pack_factor = 64.
const PACKED_ROW_LEN: usize = EMBEDDING_DIM / PACK_FACTOR;

/// Dequantized embedding matrix: all tokens expanded to f32 at init time.
pub struct EmbeddingWeights {
    /// Full dequantized embedding matrix, row-major: `[vocab_size][EMBEDDING_DIM]`.
    pub embeddings: Vec<f32>,
    /// Vocabulary size.
    pub vocab_size: usize,
}

impl EmbeddingWeights {
    /// Load and dequantize weights from an embedded NPZ byte slice.
    pub fn from_npz_bytes(data: &[u8]) -> Result<Self, WeightsError> {
        let (packed_indices, scales, vocab_size) = parse_npz(data)?;

        // Validate dimensions
        if packed_indices.len() != vocab_size * PACKED_ROW_LEN {
            return Err(WeightsError::DimensionMismatch {
                expected: vocab_size * PACKED_ROW_LEN,
                got: packed_indices.len(),
            });
        }
        if scales.len() != vocab_size {
            return Err(WeightsError::DimensionMismatch {
                expected: vocab_size,
                got: scales.len(),
            });
        }

        // Full dequantization at init
        let mut embeddings = vec![0.0f32; vocab_size * EMBEDDING_DIM];
        for token_id in 0..vocab_size {
            let scale = f16_to_f32(scales[token_id]);
            let packed_row = &packed_indices[token_id * PACKED_ROW_LEN..(token_id + 1) * PACKED_ROW_LEN];
            let emb_row = &mut embeddings[token_id * EMBEDDING_DIM..(token_id + 1) * EMBEDDING_DIM];
            dequantize_row(packed_row, scale, emb_row);
        }

        Ok(Self { embeddings, vocab_size })
    }

    /// Get the embedding vector for a given token ID (as a slice).
    #[inline]
    pub fn get_embedding(&self, token_id: usize) -> &[f32] {
        let start = token_id * EMBEDDING_DIM;
        &self.embeddings[start..start + EMBEDDING_DIM]
    }
}

/// Dequantize a single packed row into an f32 embedding.
///
/// 2-bit quantization levels: `{0, 1, 2, 3}` → centered to `{-1.5, -0.5, 0.5, 1.5}` × scale.
#[inline]
fn dequantize_row(packed: &[u8], scale: f32, output: &mut [f32]) {
    for j in 0..EMBEDDING_DIM {
        let byte_idx = j / PACK_FACTOR;
        let bit_offset = (j % PACK_FACTOR) * 2;
        let quantized = (packed[byte_idx] >> bit_offset) & 0b11;
        output[j] = (quantized as f32 - 1.5) * scale;
    }
}

/// Convert IEEE 754 half-precision (f16) to f32.
///
/// We store f16 as raw u16 bytes and convert manually to avoid
/// pulling in the entire `half` crate for this single operation.
#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            // Zero (positive or negative)
            return f32::from_bits(sign << 31);
        }
        // Subnormal f16 → normal f32
        let mut m = mantissa;
        let mut e: i32 = -14;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let f32_exp = (e + 127) as u32;
        let f32_bits = (sign << 31) | (f32_exp << 23) | (m << 13);
        f32::from_bits(f32_bits)
    } else if exponent == 31 {
        // Inf or NaN
        let f32_bits = (sign << 31) | (0xFF << 23) | (mantissa << 13);
        f32::from_bits(f32_bits)
    } else {
        // Normal
        let f32_exp = (exponent as i32 - 15 + 127) as u32;
        let f32_bits = (sign << 31) | (f32_exp << 23) | (mantissa << 13);
        f32::from_bits(f32_bits)
    }
}

// ── Minimal NPZ parser ───────────────────────────────────────────────

/// Parse the NPZ (ZIP of .npy files) to extract `packed_indices` and `scales`.
fn parse_npz(data: &[u8]) -> Result<(Vec<u8>, Vec<u16>, usize), WeightsError> {
    let mut cursor = Cursor::new(data);
    let mut packed_indices: Option<Vec<u8>> = None;
    let mut scales_raw: Option<Vec<u16>> = None;
    let mut vocab_size: usize = 0;

    loop {
        // Read ZIP local file header
        let mut sig = [0u8; 4];
        if cursor.read_exact(&mut sig).is_err() {
            break;
        }

        if sig != [0x50, 0x4B, 0x03, 0x04] {
            // Not a local file header — we're done
            break;
        }

        // Parse local file header fields
        let mut header = [0u8; 26]; // remaining 26 bytes of the local file header
        cursor.read_exact(&mut header).map_err(|e| WeightsError::Io(e.to_string()))?;

        let compression_method = u16::from_le_bytes([header[4], header[5]]);
        let mut compressed_size = u32::from_le_bytes([header[14], header[15], header[16], header[17]]) as usize;
        let mut uncompressed_size = u32::from_le_bytes([header[18], header[19], header[20], header[21]]) as usize;
        let filename_len = u16::from_le_bytes([header[22], header[23]]) as usize;
        let extra_len = u16::from_le_bytes([header[24], header[25]]) as usize;

        // Read filename
        let mut filename_bytes = vec![0u8; filename_len];
        cursor.read_exact(&mut filename_bytes).map_err(|e| WeightsError::Io(e.to_string()))?;
        let filename = String::from_utf8_lossy(&filename_bytes).to_string();

        // Read extra field
        let mut extra = vec![0u8; extra_len];
        cursor.read_exact(&mut extra).map_err(|e| WeightsError::Io(e.to_string()))?;

        // Handle ZIP64 extra fields if sizes are 0xFFFFFFFF
        if (compressed_size == 0xFFFFFFFF || uncompressed_size == 0xFFFFFFFF) && extra_len >= 4 {
            let mut offset = 0;
            while offset + 4 <= extra.len() {
                let tag = u16::from_le_bytes([extra[offset], extra[offset + 1]]);
                let block_len = u16::from_le_bytes([extra[offset + 2], extra[offset + 3]]) as usize;
                offset += 4;
                if tag == 0x0001 && offset + block_len <= extra.len() {
                    let mut zip64_offset = offset;
                    if uncompressed_size == 0xFFFFFFFF && zip64_offset + 8 <= offset + block_len {
                        uncompressed_size = u64::from_le_bytes(
                            extra[zip64_offset..zip64_offset + 8].try_into().unwrap(),
                        ) as usize;
                        zip64_offset += 8;
                    }
                    if compressed_size == 0xFFFFFFFF && zip64_offset + 8 <= offset + block_len {
                        compressed_size = u64::from_le_bytes(
                            extra[zip64_offset..zip64_offset + 8].try_into().unwrap(),
                        ) as usize;
                    }
                    break;
                }
                offset += block_len;
            }
        }

        // Read file data
        let mut raw_data = vec![0u8; compressed_size];
        cursor.read_exact(&mut raw_data).map_err(|e| WeightsError::Io(e.to_string()))?;

        let file_data = match compression_method {
            0 => raw_data,
            8 => {
                use flate2::read::DeflateDecoder;
                let mut decoder = DeflateDecoder::new(&raw_data[..]);
                let mut decompressed = Vec::with_capacity(uncompressed_size);
                decoder.read_to_end(&mut decompressed).map_err(|e| WeightsError::Io(e.to_string()))?;
                decompressed
            }
            _ => return Err(WeightsError::InvalidNpy("unsupported zip compression method")),
        };

        // Parse the .npy inside
        let name = filename.trim_end_matches(".npy");
        match name {
            "packed_indices" => {
                let (array_data, shape) = parse_npy_raw(&file_data, "u1")?;
                if shape.len() == 2 {
                    vocab_size = shape[0];
                }
                packed_indices = Some(array_data);
            }
            "scales" => {
                let (array_data, _shape) = parse_npy_raw(&file_data, "f2")?;
                // Convert raw bytes to u16 values (little-endian f16)
                let u16_vec: Vec<u16> = array_data
                    .chunks_exact(2)
                    .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                scales_raw = Some(u16_vec);
            }
            _ => {
                // Skip bits, dim, use_qjl — we know the values
            }
        }
    }

    let packed = packed_indices.ok_or(WeightsError::MissingArray("packed_indices"))?;
    let scales = scales_raw.ok_or(WeightsError::MissingArray("scales"))?;

    Ok((packed, scales, vocab_size))
}

/// Parse a .npy file header and return raw data bytes and shape.
///
/// We only need to handle simple cases: u1 (uint8) and f2 (float16).
fn parse_npy_raw(data: &[u8], _expected_dtype: &str) -> Result<(Vec<u8>, Vec<usize>), WeightsError> {
    // NPY format: \x93NUMPY + major + minor + header_len + header_str
    if data.len() < 10 || &data[0..6] != b"\x93NUMPY" {
        return Err(WeightsError::InvalidNpy("bad magic number"));
    }

    let major = data[6];
    let header_len;
    let header_start;

    if major == 1 {
        header_len = u16::from_le_bytes([data[8], data[9]]) as usize;
        header_start = 10;
    } else if major == 2 {
        if data.len() < 12 {
            return Err(WeightsError::InvalidNpy("truncated v2 header"));
        }
        header_len = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        header_start = 12;
    } else {
        return Err(WeightsError::InvalidNpy("unsupported npy version"));
    }

    let header_bytes = &data[header_start..header_start + header_len];
    let header_str = std::str::from_utf8(header_bytes)
        .map_err(|_| WeightsError::InvalidNpy("non-utf8 header"))?;

    // Parse shape from header: look for 'shape': (N, M) or 'shape': (N,) or 'shape': ()
    let shape = parse_shape_from_header(header_str)?;

    let data_start = header_start + header_len;
    let array_data = data[data_start..].to_vec();

    Ok((array_data, shape))
}

/// Extract the shape tuple from a NumPy header string.
///
/// Header format example: `{'descr': '<u1', 'fortran_order': False, 'shape': (39655, 64), }`
fn parse_shape_from_header(header: &str) -> Result<Vec<usize>, WeightsError> {
    let shape_start = header.find("'shape': (")
        .or_else(|| header.find("'shape':("))
        .ok_or(WeightsError::InvalidNpy("no shape in header"))?;

    let paren_start = header[shape_start..].find('(')
        .ok_or(WeightsError::InvalidNpy("no '(' in shape"))?;
    let paren_end = header[shape_start..].find(')')
        .ok_or(WeightsError::InvalidNpy("no ')' in shape"))?;

    let shape_str = &header[shape_start + paren_start + 1..shape_start + paren_end];
    let shape_str = shape_str.trim();

    if shape_str.is_empty() {
        return Ok(vec![]); // scalar
    }

    let dims: Result<Vec<usize>, _> = shape_str
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>())
        .collect();

    dims.map_err(|_| WeightsError::InvalidNpy("failed to parse shape dimensions"))
}

/// Errors that can occur during weight loading.
#[derive(Debug)]
pub enum WeightsError {
    Io(String),
    MissingArray(&'static str),
    InvalidNpy(&'static str),
    DimensionMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for WeightsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WeightsError::Io(msg) => write!(f, "IO error: {}", msg),
            WeightsError::MissingArray(name) => write!(f, "missing array '{}' in NPZ", name),
            WeightsError::InvalidNpy(msg) => write!(f, "invalid NPY format: {}", msg),
            WeightsError::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {}, got {}", expected, got)
            }
        }
    }
}

impl std::error::Error for WeightsError {}

impl From<io::Error> for WeightsError {
    fn from(e: io::Error) -> Self {
        WeightsError::Io(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Embed the actual weights for testing
    static WEIGHTS_DATA: &[u8] = include_bytes!("../../../../data/embeddings/turboquant_weights.npz");

    #[test]
    fn test_load_weights() {
        let w = EmbeddingWeights::from_npz_bytes(WEIGHTS_DATA).expect("failed to load weights");
        assert_eq!(w.vocab_size, 39655);
        assert_eq!(w.embeddings.len(), 39655 * EMBEDDING_DIM);
    }

    #[test]
    fn test_embedding_values() {
        let w = EmbeddingWeights::from_npz_bytes(WEIGHTS_DATA).expect("failed to load weights");

        // Verify token 100 against known reference values
        let emb = w.get_embedding(100);
        assert_eq!(emb.len(), EMBEDDING_DIM);

        // Known reference: scale = 0.9052734375
        // First value: q=0 → (0 - 1.5) * 0.905... = -1.357910...
        assert!((emb[0] - (-1.3579102)).abs() < 0.001,
            "expected ~-1.358, got {}", emb[0]);
    }

    #[test]
    fn test_bos_embedding() {
        let w = EmbeddingWeights::from_npz_bytes(WEIGHTS_DATA).expect("failed to load weights");
        let bos = w.get_embedding(0); // <s> token

        // Known reference: [0.4699707, 0.4699707, 1.4099121, -0.4699707, -0.4699707]
        assert!((bos[0] - 0.4699707).abs() < 0.001);
        assert!((bos[1] - 0.4699707).abs() < 0.001);
        assert!((bos[2] - 1.4099121).abs() < 0.001);
    }

    #[test]
    fn test_f16_conversion() {
        // 1.0 in f16 = 0x3C00
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-6);
        // 0.0 in f16 = 0x0000
        assert!((f16_to_f32(0x0000) - 0.0).abs() < 1e-6);
        // -1.0 in f16 = 0xBC00
        assert!((f16_to_f32(0xBC00) - (-1.0)).abs() < 1e-6);
    }
}
