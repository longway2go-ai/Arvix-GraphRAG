"""
pipeline/text_extractor.py
───────────────────────────
Stage 5: Extract clean text sections and run NER to identify
methods, datasets, metrics, and concepts.

Fixed: expanded regex patterns to cover CV, NLP, RL, and multimodal papers.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.models import ExtractedElement, ElementType
from pipeline.layout_detector import RawElement


# ── Section header patterns ───────────────────────────────────────────────────

SECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("abstract",      re.compile(r"\babstract\b",                          re.I)),
    ("introduction",  re.compile(r"\bintroduction\b",                      re.I)),
    ("related_work",  re.compile(r"\brelated\s+work\b",                    re.I)),
    ("methodology",   re.compile(r"\b(method(ology)?|approach|proposed)\b",re.I)),
    ("experiments",   re.compile(r"\b(experiments?|experimental\s+setup)\b",re.I)),
    ("results",       re.compile(r"\b(results?|findings)\b",               re.I)),
    ("evaluation",    re.compile(r"\bevaluation\b",                        re.I)),
    ("discussion",    re.compile(r"\bdiscussion\b",                        re.I)),
    ("conclusion",    re.compile(r"\bconclusion\b",                        re.I)),
    ("references",    re.compile(r"\breferences?\b",                       re.I)),
]

def classify_section(text: str) -> str:
    for name, pattern in SECTION_PATTERNS:
        if pattern.search(text):
            return name
    return "body"


# ── Comprehensive entity patterns ─────────────────────────────────────────────
# Covers NLP, CV, RL, multimodal, and general ML papers

DATASET_PATTERNS = re.compile(
    r"\b("
    # Vision datasets
    r"ImageNet|ImageNet[-\s]?1[Kk]|ImageNet[-\s]?21[Kk]|"
    r"COCO|MS[-\s]?COCO|"
    r"Pascal\s*VOC|VOC\s*20\d{2}|"
    r"CIFAR[-\s]?\d+|MNIST|Fashion[-\s]?MNIST|"
    r"ADE20[Kk]|Cityscapes|KITTI|nuScenes|"
    r"Places\d*|SUN\d*|LFW|CelebA|FFHQ|"
    r"LAION[-\s]?\d*[BM]?|CC\d+[BM]?|"
    r"Kinetics[-\s]?\d*|UCF\d*|HMDB\d*|"
    r"NYU[-\s]?Depth|ScanNet|ShapeNet|ModelNet\d*|"
    r"OpenImages|Objects365|LVIS|"
    # NLP datasets
    r"SQuAD[\s\d\.]*|GLUE|SuperGLUE|"
    r"WikiText[-\s]?\d*|Common\s*Crawl|BookCorpus|"
    r"WMT[-\s]?\d{2,4}|IWSLT[-\s]?\d{2,4}|"
    r"Penn\s+Treebank|CoNLL[-\s]?\d{2,4}|"
    r"SNLI|MultiNLI|SST[-\s]?\d*|IMDB|"
    r"MS\s*MARCO|Natural\s*Questions|TriviaQA|"
    # Other
    r"OpenWebText|RedPajama|Pile|"
    r"VQA[\s\dv\.]*|GQA|OK[-\s]?VQA|"
    r"RefCOCO|Flickr\d*[Ck]?"
    r")\b",
    re.IGNORECASE,
)

MODEL_PATTERNS = re.compile(
    r"\b("
    # Transformers / Attention
    r"Transformer|Attention\s+Is\s+All|"
    r"ViT|Vision\s+Transformer|DeiT|Swin\s*Transformer|Swin[-\s]?[BSLT]|"
    r"BERT|RoBERTa|DeBERTa|ALBERT|XLNet|ELECTRA|"
    r"GPT[-\s]?\d*|GPT[-\s]?[A-Z]+|ChatGPT|"
    r"T5|T5[-\s]?\w+|mT5|BART|PEGASUS|"
    r"LLaMA[-\s]?\d*|Llama[-\s]?\d*|Alpaca|Vicuna|Mistral|"
    r"CLIP|ALIGN|BLIP[-\s]?\d*|Flamingo|"
    r"DALL[-\s]?E[\s\d]*|Stable\s+Diffusion|Midjourney|"
    # CNN architectures
    r"ResNet[-\s]?\d*|ResNeXt[-\s]?\d*|"
    r"VGG[-\s]?\d*|AlexNet|ZFNet|"
    r"EfficientNet[-\s]?\w*|EfficientDet|"
    r"DenseNet[-\s]?\d*|MobileNet[-\s]?[Vv]\d*|"
    r"InceptionV?\d*|Xception|NASNet|"
    r"RegNet[-\s]?\w*|ConvNeXt[-\s]?\w*|"
    # Detection / Segmentation
    r"YOLO[-\s]?[Vv]?\d*|YOLOv\d+|"
    r"Faster\s*R[-\s]?CNN|Mask\s*R[-\s]?CNN|"
    r"RetinaNet|DETR|Deformable\s*DETR|"
    r"UNet|U[-\s]?Net|SegNet|FCN|DeepLab[-\s]?[Vv]\d*|"
    r"SAM|Segment\s+Anything|"
    # Self-supervised / contrastive
    r"SimCLR|MoCo[-\s]?[Vv]?\d*|BYOL|DINO|DINOv\d*|"
    r"MAE|BEiT|SimMIM|"
    # Attention variants (important for your papers!)
    r"NAT|Neighborhood\s+Attention|Dilated\s+.*Attention|"
    r"Window\s+Attention|Shifted\s+Window|"
    r"CAM|Grad[-\s]?CAM|Score[-\s]?CAM|"
    # Diffusion models
    r"DDPM|DDIM|Score[-\s]?SDE|"
    # GAN
    r"GAN|StyleGAN[-\s]?\d*|BigGAN|CycleGAN|Pix2Pix"
    r")\b",
    re.IGNORECASE,
)

METRIC_PATTERNS = re.compile(
    r"\b("
    r"accuracy|top[-\s]?\d+\s+accuracy|"
    r"mAP|mean\s+average\s+precision|AP\d*|AP[@\s]\d+|"
    r"mIoU|mean\s+IoU|IoU|"
    r"FID|IS|Inception\s+Score|"
    r"BLEU[-\s]?\d*|ROUGE[-\s]?\w*|METEOR|CIDEr|SPICE|"
    r"F1[-\s]?score|F[-\s]?score|precision|recall|"
    r"perplexity|PSNR|SSIM|LPIPS|"
    r"WER|CER|TER|"
    r"AUC[-\s]?ROC?|ROC|"
    r"throughput|latency|FPS|GFLOPs|FLOPs|params"
    r")\b",
    re.IGNORECASE,
)


# ── spaCy loader ──────────────────────────────────────────────────────────────

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
        logger.debug("Loaded spaCy en_core_web_sm")
    except Exception as e:
        logger.warning(f"spaCy unavailable: {e}")
        _nlp = None
    return _nlp


# ── Entity extraction ─────────────────────────────────────────────────────────

def extract_entities(text: str) -> dict[str, list[str]]:
    """
    Extract named entities from text using regex + spaCy.
    Returns dict with: methods, datasets, metrics, concepts, persons, orgs.
    """
    entities: dict[str, set[str]] = {
        "methods":  set(),
        "datasets": set(),
        "metrics":  set(),
        "concepts": set(),
        "persons":  set(),
        "orgs":     set(),
    }

    # ── Regex patterns ────────────────────────────────────────────────────────
    for m in DATASET_PATTERNS.finditer(text):
        val = m.group().strip()
        if len(val) > 2:
            entities["datasets"].add(val)

    for m in METRIC_PATTERNS.finditer(text):
        val = m.group().strip()
        if len(val) > 1:
            entities["metrics"].add(val)

    for m in MODEL_PATTERNS.finditer(text):
        val = m.group().strip()
        if len(val) > 2:
            entities["methods"].add(val)

    # ── spaCy NER ─────────────────────────────────────────────────────────────
    nlp = _get_nlp()
    if nlp and len(text) > 5:
        doc = nlp(text[:100_000])
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entities["persons"].add(ent.text.strip())
            elif ent.label_ in ("ORG", "PRODUCT"):
                entities["orgs"].add(ent.text.strip())
            elif ent.label_ in ("WORK_OF_ART", "EVENT"):
                entities["concepts"].add(ent.text.strip())

    return {k: sorted(v) for k, v in entities.items() if v}


# ── Main extractor ────────────────────────────────────────────────────────────

def extract_text_elements(
    raw_elements: list[RawElement],
    paper_id: str,
    pdf_path: str,
    save_dir: Path | None = None,
    **kwargs,
) -> list[ExtractedElement]:
    """
    Process text and title RawElements into ExtractedElement objects.
    Assigns section context and runs entity extraction on each block.
    """
    extracted: list[ExtractedElement] = []
    current_section = "body"

    for raw in raw_elements:
        text = raw.text.strip()

        if len(text) < 15:
            continue
        if re.match(r"^\[\d+\]", text) or re.match(r"^\d+\.\s+[A-Z]", text):
            continue

        # ── Title / section header ────────────────────────────────────────────
        if raw.raw_type in ("Title", "Header"):
            current_section = classify_section(text)
            extracted.append(ExtractedElement(
                element_type = ElementType.TITLE,
                content      = text,
                paper_id     = paper_id,
                page_number  = raw.page_number,
                bbox         = raw.bbox,
                metadata     = {"section": current_section},
            ))
            continue

        if current_section == "references":
            continue

        # ── Narrative text ─────────────────────────────────────────────────────
        entities = extract_entities(text)

        extracted.append(ExtractedElement(
            element_type = ElementType.TEXT,
            content      = text,
            paper_id     = paper_id,
            page_number  = raw.page_number,
            bbox         = raw.bbox,
            metadata     = {
                "section":    current_section,
                "word_count": len(text.split()),
                "entities":   entities,
            },
        ))

    methods_found  = sum(len(e.metadata.get("entities", {}).get("methods",  [])) for e in extracted)
    datasets_found = sum(len(e.metadata.get("entities", {}).get("datasets", [])) for e in extracted)

    logger.debug(
        f"[{paper_id}] Text extractor: {len(extracted)} elements "
        f"({sum(1 for e in extracted if e.element_type == ElementType.TITLE)} titles, "
        f"{methods_found} methods, {datasets_found} datasets found)"
    )
    return extracted