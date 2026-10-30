//! Content fingerprints: FNV-1a over 128 bits. A fingerprint tells whether two builds would
//! produce the same analyses or vectors; it is not a security hash. At 128 bits, two different
//! resource sets colliding by accident is out of reach, and the hash needs no dependency.

/// An FNV-1a-128 state, fed in named sections so that moving bytes from one input to another
/// changes the result.
#[derive(Clone, Copy)]
pub(crate) struct Fingerprint(u128);

impl Fingerprint {
    const OFFSET: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
    const PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;

    pub(crate) fn new() -> Self {
        Self(Self::OFFSET)
    }

    fn bytes(mut self, data: &[u8]) -> Self {
        for &b in data {
            self.0 ^= u128::from(b);
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
        self
    }

    /// Feeds a section: its name, its length and its bytes.
    pub(crate) fn section(self, name: &str, data: &[u8]) -> Self {
        self.bytes(name.as_bytes())
            .bytes(&(data.len() as u64).to_le_bytes())
            .bytes(data)
    }

    /// The fingerprint as 32 hex digits.
    pub(crate) fn hex(self) -> String {
        format!("{:032x}", self.0)
    }
}

/// The hex fingerprint of one byte string.
pub(crate) fn of(data: &[u8]) -> String {
    Fingerprint::new().section("", data).hex()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sections_are_delimited() {
        let ab = Fingerprint::new()
            .section("x", b"ab")
            .section("y", b"c")
            .hex();
        let a_bc = Fingerprint::new()
            .section("x", b"a")
            .section("y", b"bc")
            .hex();
        assert_ne!(ab, a_bc);
        assert_eq!(ab.len(), 32);
        assert_eq!(of(b"akana"), of(b"akana"));
        assert_ne!(of(b"akana"), of(b"akanA"));
    }
}
