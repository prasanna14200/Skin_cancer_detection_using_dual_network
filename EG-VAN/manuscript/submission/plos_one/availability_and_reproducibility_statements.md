# Draft statements for author verification

These are **not finalized public-availability claims**. PLOS requires a truthful [Data Availability Statement](https://journals.plos.org/plosone/s/data-availability) in its submission form and expects [code underpinning findings](https://journals.plos.org/plosone/s/materials-software-and-code-sharing) to be shareable upon publication, subject to documented legitimate restrictions.

## Data availability — conditional draft

“HAM10000 and PH² source images were obtained from their respective dataset providers under the providers' access terms. The frozen split definition, validation and final evaluation predictions/probabilities, confusion matrices, metrics and figure-underlying values supporting this article [WILL BE DEPOSITED IN AUTHOR-APPROVED REPOSITORY WITH PERSISTENT IDENTIFIER, SUBJECT TO DATASET LICENSE AND PRIVACY REVIEW]. [IF ANY DATA CANNOT BE PUBLICLY SHARED, STATE THE SPECIFIC LEGAL/ETHICAL RESTRICTION AND HOW QUALIFIED RESEARCHERS MAY ACCESS THEM.]”

Do not submit this text until the archive, contents, permissions and link exist. Local repository paths do not establish public availability.

## Code availability — conditional draft

“The author-generated preprocessing, model, numerical-policy, training and final-evaluation code underpinning the findings [WILL BE AVAILABLE AT VERIFIED PERMANENT REPOSITORY URL/DOI AND LICENSE]. Dependencies, frozen configuration and split hashes will accompany the archive. [STATE ANY JUSTIFIED RESTRICTION AND ACCESS ROUTE IF APPLICABLE.]”

## Reproducibility / model provenance — evidence-supported draft

“The reported final results use the recovered Stage 15 selected epoch-16 checkpoint, SHA256 `85fcad4b184da16dd741f2d539a05f8a1f0c8d1d015298f4ce5aadf7ef4d16d5`. Bounded continuation from the archived epoch-13 state reproduced the available epoch-14–16 training-history and validation observables exactly. The original epoch-16 weight file did not persist, so byte identity to that missing file cannot be established. The frozen HAM split SHA256 is `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883`. The selective-FP32 policy applied only to `resnet.nonlocal3` q@k affinity; this addressed FP16 overflow and is not evidence of an accuracy gain.”

## Checkpoint availability — conditional draft

“The recovered selected checkpoint [WILL BE DEPOSITED AT VERIFIED LINK/DOI, IF AUTHOR-APPROVED AND PERMITTED]. [IF NOT RELEASED, STATE A POLICY-COMPLIANT REASON AND ACCESS ARRANGEMENT; CONFIRM WITH THE EDITOR WHETHER THE MODEL FILE IS REQUIRED.]”
