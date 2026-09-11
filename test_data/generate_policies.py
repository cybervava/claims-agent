"""Generate sample insurance policy PDFs for RAG testing.

Usage:
    pip install reportlab
    python test_data/generate_policies.py

Writes four Meridian policy wordings to test_data/policies/*.pdf.
Each policy has the same section structure as sample_data/policies/*.md
(period, perils, exclusions, excess, limits, claim conditions, fraud) but
distinct products, policy-number prefixes and figures, so retrieval tests
can check that the right document is cited.
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT_DIR = Path(__file__).parent / "policies"

# Each policy: (filename, title, subtitle, [(section heading, [clauses])])
POLICIES: list[tuple[str, str, str, list[tuple[str, list[str]]]]] = [
    (
        "travel_single_trip_policy.pdf",
        "Meridian Travel Single-Trip Policy",
        "Policy Wording v1.4",
        [
            ("Section 1 — Policy Period and Eligibility", [
                "1.1 Cover applies to one return trip commencing and ending in the policyholder's country of residence, with a maximum trip duration of 45 days.",
                "1.2 The policyholder must be under 70 years of age at the date of purchase and must purchase the policy before departure.",
                "1.3 Policy numbers for this product are in the format TR-XXXXXXX (e.g. TR-3310582).",
                "1.4 Cancellation cover begins on the date the policy is purchased; all other sections begin on the departure date shown on the schedule.",
            ]),
            ("Section 2 — Covered Events", [
                "2.1 Cancellation or curtailment of the trip due to the death, serious injury or serious illness of the policyholder, a travelling companion or a close relative.",
                "2.2 Emergency medical and dental treatment received abroad for an illness or injury first occurring during the trip.",
                "2.3 Emergency repatriation to the country of residence when medically necessary and approved by the 24-hour assistance line.",
                "2.4 Loss, theft or damage of personal baggage and personal money while on the trip.",
                "2.5 Delayed baggage: essential replacement items where checked baggage is delayed more than 12 hours on the outward journey.",
                "2.6 Missed departure due to the failure of scheduled public transport or a road accident on the way to the departure point.",
                "2.7 Travel delay of more than 12 hours to the first outward or final return departure caused by strike, adverse weather or mechanical breakdown of the carrier.",
            ]),
            ("Section 3 — Exclusions", [
                "3.1 Any pre-existing medical condition not declared to and accepted in writing by the insurer before the policy was issued.",
                "3.2 Claims arising from travel against the advice of a medical practitioner, or where the purpose of the trip is to obtain medical treatment.",
                "3.3 Claims arising from the policyholder being under the influence of alcohol or drugs.",
                "3.4 Winter sports, scuba diving below 30 metres, mountaineering requiring ropes, and motorcycling over 125cc, unless the Adventure Activities extension is shown on the schedule.",
                "3.5 Baggage or money left unattended, left in a vehicle overnight, or left in a public place.",
                "3.6 Valuables carried in checked baggage.",
                "3.7 Cancellation caused by a change of mind, disinclination to travel or financial circumstances other than redundancy.",
                "3.8 Travel to any country or region against which the government of the country of residence has advised against all travel.",
                "3.9 Any claim where the incident occurred before the trip departure date (other than cancellation) or after the trip return date.",
            ]),
            ("Section 4 — Deductible (Excess)", [
                "4.1 An excess of 75 applies per person per incident under the cancellation, medical and baggage sections.",
                "4.2 No excess applies to travel delay, missed departure or delayed baggage benefits.",
                "4.3 Where the Excess Waiver extension is shown on the schedule, no excess applies to any section.",
            ]),
            ("Section 5 — Limits and Sub-limits", [
                "5.1 Cancellation or curtailment: up to 5,000 per person.",
                "5.2 Emergency medical expenses abroad: up to 2,000,000 per person, including repatriation.",
                "5.3 Emergency dental treatment for pain relief only: sub-limit 350 per person.",
                "5.4 Personal baggage: up to 2,000 per person; single-item limit 300; valuables total limit 400.",
                "5.5 Personal money: up to 300 per person, of which cash is limited to 200.",
                "5.6 Delayed baggage: up to 150 per person after 12 hours.",
                "5.7 Travel delay: 25 per full 12-hour period, up to 250 per person.",
                "5.8 Missed departure: up to 500 per person for additional travel and accommodation.",
            ]),
            ("Section 6 — Claim Conditions", [
                "6.1 The 24-hour assistance line must be contacted before any hospital admission, and before any repatriation or curtailment is arranged, or benefit may be limited to what would have been paid had it been contacted.",
                "6.2 Claims must be submitted within 31 days of returning from the trip.",
                "6.3 Theft or loss of baggage or money must be reported to the local police within 24 hours of discovery and a written police report obtained.",
                "6.4 Loss or damage of baggage in the custody of an airline or carrier must be reported to that carrier before leaving the airport and a Property Irregularity Report obtained.",
                "6.5 Cancellation claims require a medical certificate from the treating doctor confirming the patient was unfit to travel, and written confirmation from the travel provider of the non-refundable amount.",
                "6.6 Travel delay claims require written confirmation from the carrier of the scheduled and actual departure times and the reason for the delay.",
                "6.7 Original receipts are required for all items claimed above 50; items without receipts are settled at the insurer's discretion.",
                "6.8 Baggage items are settled at market value with an allowance for wear and tear; items less than 12 months old are settled at replacement cost.",
            ]),
            ("Section 7 — Fraud", [
                "7.1 If any claim is fraudulent or exaggerated, or any false document is supplied, all benefit under this policy is forfeited and the policy is cancelled from the date of the fraudulent act.",
            ]),
        ],
    ),
    (
        "pet_health_policy.pdf",
        "Meridian Pet Health Lifetime Policy",
        "Policy Wording v2.3",
        [
            ("Section 1 — Policy Period and Eligibility", [
                "1.1 Cover applies to the dog or cat named on the schedule for a rolling 12-month policy period, renewable annually for the life of the pet provided cover is not lapsed.",
                "1.2 The pet must be between 8 weeks and 8 years of age at first inception and must be microchipped and registered at the insured address.",
                "1.3 Policy numbers for this product are in the format PH-XXXXXXX (e.g. PH-7720419).",
                "1.4 A waiting period of 14 days applies to illness cover and 48 hours to accident cover from the inception date. Conditions first showing symptoms during the waiting period are treated as pre-existing.",
            ]),
            ("Section 2 — Covered Treatment", [
                "2.1 Veterinary fees for the treatment of an accidental injury or illness first occurring during the policy period.",
                "2.2 Diagnostic tests, imaging, surgery, hospitalisation, prescribed medication and follow-up consultations relating to a covered condition.",
                "2.3 Complementary treatment (physiotherapy, hydrotherapy, acupuncture) when referred by a vet for a covered condition.",
                "2.4 Dental treatment resulting from an accident. Dental illness is covered only where a dental check has been carried out within the 12 months before the claim.",
                "2.5 Third-party liability for injury or property damage caused by an insured dog.",
                "2.6 Advertising and reward costs if the pet is lost or stolen, and the purchase price if not recovered within 30 days.",
                "2.7 Death from injury or illness before the pet's 9th birthday: the purchase price of the pet.",
            ]),
            ("Section 3 — Exclusions", [
                "3.1 Pre-existing conditions: any illness or injury, or symptom of one, that occurred or was noticed before the inception date or during the waiting period, and any related or bilateral condition.",
                "3.2 Routine and preventative treatment: vaccinations, flea and worm control, neutering, nail clipping, grooming and routine dental scaling.",
                "3.3 Pregnancy, breeding, whelping and any complications thereof.",
                "3.4 Behavioural conditions unless treated by a referred veterinary behaviourist and shown as covered on the schedule.",
                "3.5 Prescription diets and food, except for 50% of the cost when prescribed for a covered urinary or digestive condition for a maximum of 3 months.",
                "3.6 Treatment for conditions arising from a failure to follow veterinary advice or to keep vaccinations current.",
                "3.7 Dogs of a breed listed under the Dangerous Dogs Act of the country of residence, or used for racing, guarding or commercial breeding.",
                "3.8 Costs incurred after the policy has lapsed or been cancelled, even where the condition began during the policy period.",
            ]),
            ("Section 4 — Deductible (Excess)", [
                "4.1 A fixed excess of 99 applies once per condition per policy year.",
                "4.2 For pets aged 8 years or over at the start of the policy year, a co-payment of 20% of the remaining claim applies in addition to the fixed excess.",
                "4.3 No excess applies to the third-party liability, advertising and reward, or death benefit sections.",
            ]),
            ("Section 5 — Limits and Sub-limits", [
                "5.1 Veterinary fees: up to 7,000 per policy year, refreshing at each renewal. Unused amounts do not carry forward.",
                "5.2 Complementary treatment: sub-limit 750 per policy year within the veterinary fees limit.",
                "5.3 Dental treatment: sub-limit 1,000 per policy year within the veterinary fees limit.",
                "5.4 Third-party liability (dogs only): up to 1,000,000 per incident.",
                "5.5 Advertising and reward: up to 250; purchase price if not found: up to 1,500.",
                "5.6 Death benefit: purchase price up to 1,500, reducing to 0 from the pet's 9th birthday.",
                "5.7 Emergency boarding fees if the policyholder is hospitalised for more than 4 consecutive days: up to 500 per policy year.",
            ]),
            ("Section 6 — Claim Conditions", [
                "6.1 Claims must be submitted within 90 days of the treatment date on the insurer's claim form, with the treating vet's section completed.",
                "6.2 A full clinical history from every practice that has treated the pet is required for the first claim on each new condition.",
                "6.3 Itemised invoices are required. Where the vet's fees are, in the insurer's reasonable opinion, higher than customary in the area, the insurer may settle at the customary amount.",
                "6.4 For the lost or stolen pet benefit, the loss must be reported to the local police, the microchip database and a local animal shelter within 48 hours.",
                "6.5 Third-party liability incidents must be reported within 7 days and no admission of liability may be made without the insurer's written consent.",
                "6.6 A post-mortem report may be requested before the death benefit is paid.",
                "6.7 Direct payment to the vet is available only where the practice has agreed in advance and the claim exceeds 300.",
            ]),
            ("Section 7 — Fraud", [
                "7.1 Fraudulent or exaggerated claims, including altered invoices or clinical records, void all benefit under this policy and the policy is cancelled from the date of the fraudulent act.",
            ]),
        ],
    ),
    (
        "gadget_electronics_policy.pdf",
        "Meridian Gadget & Personal Electronics Policy",
        "Policy Wording v1.1",
        [
            ("Section 1 — Policy Period and Insured Items", [
                "1.1 Cover applies to the gadgets listed on the schedule, each identified by make, model and serial or IMEI number, for a 12-month policy period.",
                "1.2 Eligible gadgets are mobile phones, tablets, laptops, smart watches, headphones, cameras and e-readers that are less than 36 months old and in full working order at inception.",
                "1.3 Policy numbers for this product are in the format GD-XXXXXXX (e.g. GD-5502173).",
                "1.4 Cover is worldwide for up to 90 days in any policy year outside the country of residence.",
            ]),
            ("Section 2 — Covered Perils", [
                "2.1 Accidental damage, including cracked screens, drops and impact damage.",
                "2.2 Liquid damage, including submersion and spillage.",
                "2.3 Theft of the gadget, including theft from a locked vehicle where the gadget was concealed from view and the vehicle shows evidence of forced entry.",
                "2.4 Loss of the gadget, where the Loss extension is shown on the schedule for that item (mobile phones and smart watches only).",
                "2.5 Mechanical or electrical breakdown occurring after the manufacturer's warranty has expired.",
                "2.6 Unauthorised calls, data usage and in-app purchases following theft or loss of a mobile phone, from the time of the incident until the network provider is notified.",
                "2.7 Accessories (cases, chargers, straps) purchased for and damaged, stolen or lost together with the insured gadget.",
            ]),
            ("Section 3 — Exclusions", [
                "3.1 Cosmetic damage that does not affect the functionality of the gadget, including scratches, dents and worn finishes.",
                "3.2 Theft or loss where the gadget was left unattended in a public place, or left in an unlocked vehicle or building.",
                "3.3 Gadgets left in a vehicle overnight between 21:00 and 07:00.",
                "3.4 Damage caused by wilful neglect, deliberate act, or use contrary to the manufacturer's instructions.",
                "3.5 Damage arising from unauthorised repair, modification, jailbreaking or the installation of unofficial firmware.",
                "3.6 Loss of or damage to data, software or downloaded content, and the cost of data recovery.",
                "3.7 Breakdown covered by the manufacturer's warranty, a recall, or any other guarantee.",
                "3.8 Any gadget whose serial or IMEI number cannot be verified or does not match the schedule.",
                "3.9 Gadgets used for business hire or lent to any person not living at the insured address.",
            ]),
            ("Section 4 — Deductible (Excess)", [
                "4.1 The excess per claim depends on the replacement value of the gadget: 25 for items up to 250; 50 for items from 251 to 750; 75 for items from 751 to 1,500; 100 for items above 1,500.",
                "4.2 An additional excess of 50 applies to every loss claim under the Loss extension.",
                "4.3 A separate excess applies to each gadget claimed in the same incident.",
            ]),
            ("Section 5 — Limits and Sub-limits", [
                "5.1 The maximum payable for any single gadget is the lower of its replacement value with an equivalent model and the sum insured shown on the schedule for that item, up to 2,500.",
                "5.2 Accessories: sub-limit 150 per claim.",
                "5.3 Unauthorised use following theft or loss: sub-limit 1,000 per claim, and 100 per day where the network is not notified within 24 hours.",
                "5.4 Maximum 3 claims per gadget per policy year; a fourth claim results in that gadget being removed from cover at renewal.",
                "5.5 Where a gadget is replaced, the replacement becomes the insured item and the sum insured is unchanged.",
            ]),
            ("Section 6 — Claim Conditions", [
                "6.1 Claims must be notified within 14 days of the incident, or of discovery in the case of theft or loss.",
                "6.2 Theft and loss must be reported to the police within 48 hours and a crime reference or lost-property reference supplied; mobile phone thefts must also be reported to the network provider within 24 hours to block the handset.",
                "6.3 Proof of purchase showing the date, price and serial or IMEI number is required for every claim. Second-hand gadgets are not covered without a dated receipt from a retailer.",
                "6.4 Damaged gadgets must be sent to the insurer's approved repairer for assessment before any repair is authorised. Repairs by any other repairer are not reimbursed.",
                "6.5 If a gadget is beyond economic repair, the insurer will replace it with the same model or, if unavailable, a model of equivalent specification. Cash settlement is offered only where no equivalent model is available.",
                "6.6 The policyholder must remove SIM cards, memory cards and personal data before sending a gadget for repair; the insurer is not responsible for data on devices in its custody.",
                "6.7 Lost or stolen gadgets that are later recovered must be returned to the insurer if a replacement has been provided.",
            ]),
            ("Section 7 — Fraud", [
                "7.1 If any claim is fraudulent or exaggerated, or if a false serial or IMEI number, receipt or police reference is supplied, all benefit under this policy is forfeited and the policy is cancelled from the date of the fraudulent act.",
            ]),
        ],
    ),
    (
        "small_business_property_policy.pdf",
        "Meridian Small Business Property & Interruption Policy",
        "Policy Wording v4.0",
        [
            ("Section 1 — Policy Period and Insured Premises", [
                "1.1 Cover applies to the business premises shown on the schedule, and to the business contents, stock and equipment owned by the insured business or for which it is responsible, for a 12-month policy period.",
                "1.2 The insured business must have an annual turnover below 2,000,000 and no more than 25 employees at any one premises.",
                "1.3 Policy numbers for this product are in the format BP-XXXXXXX (e.g. BP-9106340).",
                "1.4 Business interruption cover is subject to the indemnity period shown on the schedule, being 12 months unless otherwise stated.",
            ]),
            ("Section 2 — Covered Perils", [
                "2.1 Fire, smoke, lightning, explosion and earthquake.",
                "2.2 Storm, flood and escape of water from any fixed water, heating or sprinkler installation.",
                "2.3 Theft or attempted theft involving forcible and violent entry to or exit from the premises, and hold-up or robbery of money within the premises.",
                "2.4 Riot, civil commotion, malicious damage and vandalism.",
                "2.5 Impact by vehicle, aircraft or falling trees; collapse of aerials and satellite dishes.",
                "2.6 Accidental damage to fixed glass, signs and shop fronts.",
                "2.7 Deterioration of refrigerated stock following breakdown of refrigeration equipment or failure of the public electricity supply for more than 4 hours.",
                "2.8 Business interruption: loss of gross profit and increased cost of working resulting from an insured peril under 2.1 to 2.7 damaging the premises or contents.",
                "2.9 Denial of access: loss of gross profit where access to the premises is prevented for more than 24 hours by damage to neighbouring property or by order of a public authority following an insured peril within 1 kilometre.",
            ]),
            ("Section 3 — Exclusions", [
                "3.1 Theft not involving forcible and violent entry or exit, including theft by employees, and theft from any part of the premises open to the public unless by hold-up.",
                "3.2 Theft of stock or contents from vehicles, or from the open yard, unless secured in a locked steel container.",
                "3.3 Loss or damage occurring while the premises have been unoccupied for more than 30 consecutive days, unless the insurer has been notified and agreed continued cover in writing.",
                "3.4 Gradual deterioration, wear and tear, rust, corrosion, mould, wet or dry rot, and defective design or workmanship.",
                "3.5 Electrical or mechanical breakdown of any machine, other than as provided under 2.7 for refrigerated stock.",
                "3.6 Loss of money exceeding 1,000 in the premises outside business hours unless held in a locked safe, and money in transit exceeding 2,500 per carrying.",
                "3.7 Business interruption losses arising from failure of utilities, telecommunications or internet services except as provided under 2.7.",
                "3.8 Losses arising from notifiable disease, pollution, contamination, terrorism or war.",
                "3.9 Fines, penalties, liquidated damages or contractual penalties incurred by the insured business.",
            ]),
            ("Section 4 — Deductible (Excess)", [
                "4.1 A standard excess of 500 applies to each property damage claim.",
                "4.2 An escape-of-water excess of 1,000 applies to each claim under 2.2 involving water escaping from installations.",
                "4.3 A theft excess of 750 applies to each theft claim.",
                "4.4 A time excess of 48 hours applies to business interruption claims; no gross profit is payable for the first 48 hours of interruption.",
                "4.5 A subsidence, heave and landslip excess of 2,500 applies where that cover is shown on the schedule.",
            ]),
            ("Section 5 — Limits and Sub-limits", [
                "5.1 Buildings, contents and stock sums insured are stated on the schedule. If at the time of loss the sum insured is less than 85% of the full reinstatement value, the claim is reduced proportionately (average).",
                "5.2 Seasonal stock increase: stock sum insured increases by 25% during November and December and for 30 days before any religious festival.",
                "5.3 Money on the premises during business hours: 3,000; in a locked safe outside business hours: 5,000; in transit: 2,500.",
                "5.4 Refrigerated stock: sub-limit 2,500 per claim.",
                "5.5 Glass, signs and shop fronts: sub-limit 5,000 per claim.",
                "5.6 Business interruption: gross profit up to the sum insured on the schedule for the indemnity period; denial of access sub-limit 25,000 or 10% of the gross profit sum insured, whichever is less.",
                "5.7 Loss of documents and computer records: sub-limit 10,000 for the cost of re-creation.",
                "5.8 Tenant's improvements: 10% of the contents sum insured where the insured business is a tenant.",
            ]),
            ("Section 6 — Claim Conditions", [
                "6.1 Incidents must be notified to the insurer within 7 days of occurrence or discovery.",
                "6.2 Theft, attempted theft, hold-up, riot and malicious damage must be reported to the police within 24 hours and a crime reference supplied.",
                "6.3 The insured must maintain and produce on request stock records, purchase invoices, sales records and audited or management accounts for the 3 years before the loss.",
                "6.4 Business interruption claims must be supported by monthly turnover figures for the 12 months before the loss and for the indemnity period, and by evidence of increased costs incurred.",
                "6.5 Property damage claims above 5,000 will be inspected by a loss adjuster appointed by the insurer before any repair or replacement is authorised; the insured must retain damaged items for inspection.",
                "6.6 The insured must comply with the security conditions on the schedule: intruder alarm maintained and set outside business hours, and all external doors fitted with the specified locks. Failure to comply voids theft cover.",
                "6.7 Settlement of buildings and contents is on a reinstatement basis; stock is settled at cost price or net realisable value, whichever is lower.",
                "6.8 Claims for refrigerated stock require a refrigeration engineer's report or confirmation from the electricity supplier of the outage duration.",
            ]),
            ("Section 7 — Fraud", [
                "7.1 If any claim is fraudulent or exaggerated, or if any false declaration, invoice or record is supplied in support of a claim, all benefit under this policy is forfeited and the policy is cancelled from the date of the fraudulent act.",
            ]),
        ],
    ),
]


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontSize=18, spaceAfter=2 * mm),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"], fontSize=10, textColor="#555555", spaceAfter=8 * mm),
        "heading": ParagraphStyle("heading", parent=base["Heading2"], fontSize=12, spaceBefore=5 * mm, spaceAfter=2 * mm),
        "clause": ParagraphStyle("clause", parent=base["Normal"], fontSize=10, leading=14, alignment=TA_JUSTIFY, spaceAfter=2 * mm),
        "footer": ParagraphStyle("footer", parent=base["Normal"], fontSize=8, textColor="#777777"),
    }


def render_policy(path: Path, title: str, subtitle: str, sections: list[tuple[str, list[str]]]) -> None:
    styles = build_styles()
    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm, topMargin=20 * mm, bottomMargin=20 * mm,
        title=title, author="Meridian Insurance (sample test data)",
    )
    story = [Paragraph(title, styles["title"]), Paragraph(f"{subtitle} · Sample document for testing only", styles["subtitle"])]
    for heading, clauses in sections:
        story.append(Paragraph(heading, styles["heading"]))
        story.extend(Paragraph(clause, styles["clause"]) for clause in clauses)
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph("This is fictitious sample policy wording generated for software testing. It is not a contract of insurance.", styles["footer"]))
    doc.build(story)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, title, subtitle, sections in POLICIES:
        render_policy(OUT_DIR / filename, title, subtitle, sections)
        print(f"wrote {OUT_DIR / filename}")


if __name__ == "__main__":
    main()
