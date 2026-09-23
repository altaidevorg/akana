//! Embedding-based semantic prototype scorer and micro-head for Turkish PII.
//!
//! Utilizes Akana's bundled 256-dimensional Model2Vec TurboQuant embedding engine
//! to compute semantic similarity against sensitive domain prototypes (Health, Credentials,
//! Financial, Address) and disambiguate polysemous or open-vocabulary expressions.

use crate::embeddings::{cosine_similarity, TurkishEmbeddings};
use std::sync::Arc;

/// Semantic category prototypes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PiiPrototypeCategory {
    Health,
    Financial,
    Credentials,
    Address,
    PrivateDate,
    PublicContext,
    PersonAgent,
    NatureObject,
}

/// Semantic prototype scorer leveraging precomputed centroid vectors.
pub struct PiiEmbeddingScorer {
    embeddings: Arc<TurkishEmbeddings>,
    health_centroid: Vec<f32>,
    financial_centroid: Vec<f32>,
    credentials_centroid: Vec<f32>,
    address_centroid: Vec<f32>,
    private_date_centroid: Vec<f32>,
    public_context_centroid: Vec<f32>,
    person_agent_centroid: Vec<f32>,
    nature_object_centroid: Vec<f32>,
}

impl PiiEmbeddingScorer {
    /// Initializes the scorer using a shared or new `TurkishEmbeddings` instance.
    pub fn new(embeddings: Arc<TurkishEmbeddings>) -> Self {
        // Pre-compute normalized centroid vectors for key sensitive semantic spaces
        let health_centroid = embeddings
            .embed("hastalık tanı teşhis tedavi ilaç kanser diyabet ameliyat tahlil sağlık raporu");
        let financial_centroid =
            embeddings.embed("maaş aylık gelir net kazanç borç bakiye para tutarı hesap bakiyesi");
        let credentials_centroid = embeddings
            .embed("şifre parola gizli pin kodu kimlik doğrulama giriş şifresi secret key");
        let address_centroid = embeddings
            .embed("ikametgah ev adresi teslimat adresi mahalle cadde sokak numara daire ilçe");
        let private_date_centroid = embeddings.embed(
            "doğum günü randevu muayene ameliyat fatura kesim teslimat abonelik sözleşme mezuniyet başvuru işe giriş rezervasyon"
        );
        let public_context_centroid = embeddings.embed(
            "kampanya indirim duyuru genelge resmi tatil festival konser seminer konferans basın bülteni mevzuat kanun"
        );
        let person_agent_centroid = embeddings.embed(
            "insan kişi vatandaş müşteri çalışan hasta hekim avukat müvekkil öğrenci yetkili müdür uzman"
        );
        let nature_object_centroid = embeddings.embed(
            "deniz göl nehir su dağ kaya taş toprak tarla bahçe gül çiçek ağaç yağmur rüzgar fırtına bulut hava"
        );

        Self {
            embeddings,
            health_centroid,
            financial_centroid,
            credentials_centroid,
            address_centroid,
            private_date_centroid,
            public_context_centroid,
            person_agent_centroid,
            nature_object_centroid,
        }
    }

    /// Access underlying TurkishEmbeddings instance.
    pub fn embeddings(&self) -> &TurkishEmbeddings {
        &self.embeddings
    }

    /// Computes cosine similarity between a candidate text span and a target prototype category.
    pub fn score_similarity(&self, text: &str, category: PiiPrototypeCategory) -> f32 {
        let vec = self.embeddings.embed(text);
        let centroid = match category {
            PiiPrototypeCategory::Health => &self.health_centroid,
            PiiPrototypeCategory::Financial => &self.financial_centroid,
            PiiPrototypeCategory::Credentials => &self.credentials_centroid,
            PiiPrototypeCategory::Address => &self.address_centroid,
            PiiPrototypeCategory::PrivateDate => &self.private_date_centroid,
            PiiPrototypeCategory::PublicContext => &self.public_context_centroid,
            PiiPrototypeCategory::PersonAgent => &self.person_agent_centroid,
            PiiPrototypeCategory::NatureObject => &self.nature_object_centroid,
        };
        cosine_similarity(&vec, centroid)
    }

    /// Evaluates whether candidate text belongs to a sensitive prototype category above a threshold.
    pub fn is_category_match(
        &self,
        text: &str,
        category: PiiPrototypeCategory,
        threshold: f32,
    ) -> bool {
        self.score_similarity(text, category) >= threshold
    }

    /// Determines if a context window around a candidate date indicates a private personal date
    /// (e.g. appointment, birth, delivery, billing) rather than a public announcement/campaign.
    pub fn is_private_date_context(&self, context: &str) -> bool {
        let vec = self.embeddings.embed(context);
        let sim_priv = cosine_similarity(&vec, &self.private_date_centroid);
        let sim_pub = cosine_similarity(&vec, &self.public_context_centroid);

        // Positive private score that exceeds public context score or has strong absolute match
        sim_priv > sim_pub && sim_priv >= 0.15 || sim_priv >= 0.35
    }

    /// Determines if a context window around a polysemous word (e.g. Deniz, Barış, Kaya, Gül)
    /// describes a person / human agent rather than a natural object or common noun.
    pub fn is_person_context(&self, context: &str) -> bool {
        let vec = self.embeddings.embed(context);
        let sim_person = cosine_similarity(&vec, &self.person_agent_centroid);
        let sim_nature = cosine_similarity(&vec, &self.nature_object_centroid);

        sim_person >= sim_nature
    }

    /// Ultra-lightweight linear classification head: W (num_classes, 256) * x + b.
    ///
    /// Computes class logits directly in pure Rust in < 1 microsecond per token vector.
    pub fn predict_micro_head(
        weights: &[f32],
        bias: &[f32],
        num_classes: usize,
        token_vec: &[f32],
    ) -> Vec<f32> {
        let dim = token_vec.len();
        let mut logits = vec![0.0f32; num_classes];

        for (c, item) in logits.iter_mut().enumerate() {
            let mut dot = bias.get(c).copied().unwrap_or(0.0);
            let row_offset = c * dim;
            for i in 0..dim {
                dot += weights[row_offset + i] * token_vec[i];
            }
            *item = dot;
        }

        logits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prototype_similarity() {
        let emb = Arc::new(TurkishEmbeddings::new());
        let scorer = PiiEmbeddingScorer::new(emb);

        let health_score =
            scorer.score_similarity("kemoterapi tedavisi görüyor", PiiPrototypeCategory::Health);
        let non_health_score =
            scorer.score_similarity("yarın futbol maçı var", PiiPrototypeCategory::Health);
        assert!(health_score > non_health_score);
        assert!(health_score > 0.30);

        let cred_score =
            scorer.score_similarity("gizli parola ve pin", PiiPrototypeCategory::Credentials);
        let non_cred_score =
            scorer.score_similarity("güzel bir hava", PiiPrototypeCategory::Credentials);
        assert!(cred_score > non_cred_score);

        // Test private date vs public context
        assert!(scorer.is_private_date_context("muayene randevum için kayıt açıldı"));
        assert!(!scorer.is_private_date_context("bayram kampanyası duyurusu kapsamında indirim"));

        // Test person context vs nature
        assert!(scorer.is_person_context("Barış Bey toplantıda yeni projeyi anlattı"));
        assert!(!scorer.is_person_context("yaz tatilinde deniz kenarında yürüyüş yaptık"));
    }
}
