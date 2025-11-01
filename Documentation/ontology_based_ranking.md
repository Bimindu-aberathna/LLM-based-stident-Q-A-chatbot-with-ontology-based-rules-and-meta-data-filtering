**. Introduction**

This section presents the ontology-based document prioritization module, a critical component of the RAG system designed to address the limitations non-academic information. While semantic similarity measures effectively capture the conceptual alignment between queries and documents, it operates independently of organizational hierarchies and institutional authority structures. Consequently, a university-wide policy and a department-specific guideline addressing the same subject matter may exhibit comparable semantic similarity scores despite fundamental differences in their applicability and authoritative relevance to individual students.

The prioritization module integrates two complementary sub-modules: hierarchical scoring and freshness-based scoring. The hierarchical scoring sub-module exploits the natural organizational structure inherent to academic institutions, recognizing that information authority and relevance correlate with organizational proximity to the student. The freshness-based scoring sub-module incorporates temporal considerations, acknowledging that institutional policies and procedures evolve over time, rendering recent documents more likely to reflect current regulations and requirements. Through the integration of these mechanisms within a unified prioritization framework, the module ensures that students receive contextually appropriate, temporally relevant, and authoritatively applicable information tailored to their specific organizational position within the institution.

Academic institutions characteristically organize themselves through hierarchical structures that reflect administrative responsibility, academic governance, and resource allocation. The typical organizational architecture follows a multi-tiered model wherein the university operates as the overarching entity, encompassing multiple faculties, which in turn contain various departments, ultimately serving individual students. This hierarchical arrangement represents more than mere administrative convenience; it embodies fundamental relationships of policy authority, regulatory scope, and information applicability.

Within this framework, each organizational level possesses the authority to issue policies, guidelines, and regulations that govern its domain. Critically, these documents exist in a hierarchical relationship wherein more specific policies issued at lower organizational levels often override, refine, or provide exceptions to general policies promulgated at higher levels. A department-specific regulation may establish requirements that supersede or supplement faculty-level guidelines, which themselves may modify university-wide policies. This cascading authority structure creates a complex information landscape where document relevance cannot be determined solely through semantic content analysis.

- **Authority and Freshness Scoring**

The prioritization process uses a weighted scoring model comprising two key factors:

1. **Institutional Hierarchical Authority:** Documents are given scores that indicate the level of relevance and authority based on their origin. Policies issued by departments or faculties, which are often more directly applicable to a student’s academic experience, are prioritized over broader university-wide documents. The importance of each document is assessed in the context of its alignment with the student's specific academic setting and the organizational structures within the university (Knollmeyer et al., 2024).
1. **Document Freshness:** The dates of publication and validity of documents are used to determine their temporal relevance.  When it comes to prioritizing, authority typically takes precedence over freshness, despite the fact that freshness offers a valuable dynamic element where more recent and relevant documents take precedence over older ones (Bansal et al., 2019).  With freshness being utilized to break ties when papers of similar hierarchical levels are in competition, this balance ensures that authoritative texts continue to take precedence.  The freshness score aligns with temporal scoring approaches used in Information Retrieval (IR) literature (Khan et al., 2018).

Together, these scoring elements produce an overall document priority score that directs response generation and retrieval.

- **Conflict Resolution and Priority Cascading**

In academic institutions, where various administrative documents may present conflicting directives or information, the ontology-based system implements conflict resolution through cascading priority rules.  The guidelines reflect the standard decision-making processes typically seen in a university setting. At every level of the hierarchy, newer documents take precedence over older versions or announcements. This temporal ordering ensures that users receive the most current and institutionally sanctioned guidance.

**Implementation of the Prioritization Module**

The prioritization module operates as an integrated component within the RAG retrieval pipeline, positioned after semantic similarity filtering and metadata-based applicability filtering. This strategic placement reflects its role in refining already-relevant results rather than performing initial retrieval. Documents reaching the prioritization module have been established as both semantically relevant to the query and applicable to the student based on metadata constraints (Lewis et al., 2020).

The implementation follows a systematic multi-stage process:

**Stage 1: Hierarchical Level Classification**

Each document undergoes classification based on metadata analysis and student organizational affiliation. The system examines document metadata fields including `hierarchy_level`, `departments_affecting`, and `faculties_affecting` to determine appropriate classification. Documents explicitly tagged with hierarchical indicators are classified accordingly. For documents lacking explicit designation, the system infers level by analyzing organizational scope metadata. Documents listing specific departments in their applicability scope are classified as department-level; those specifying faculty applicability as faculty-level; and those with broad or unspecified scope as university-level.

**Stage 2: Hierarchical Score Assignment**

Following classification, the system assigns numerical scores reflecting organizational proximity. The scoring function implements a stratified approach with calibrated differentials between levels. Department-level documents applicable to the student's department receive maximal hierarchical scores, reflecting their direct applicability. Faculty-level documents applicable to the student's faculty receive intermediate scores. University-level documents receive baseline scores. The magnitude of score differentials is calibrated such that hierarchical level dominates the final ranking while allowing secondary factors to influence outcomes when hierarchical applicability is equivalent (Guarino et al., 2009).

**Stage 3: Temporal Analysis and Freshness Scoring**

Document upload timestamps are analyzed to determine age. The system applies a graduated temporal decay function wherein scores degrade progressively with increasing document age. Recent documents (0-30 days) receive maximal freshness scores. Moderately aged documents (31-90 days) receive elevated but reduced scores. Documents aged 91-180 days receive intermediate scores, while those aged 181-365 days receive low scores. Documents exceeding one year receive minimal but non-zero scores, acknowledging possible continued validity despite increased obsolescence risk. This graduated approach creates meaningful differentiation between substantially different ages while avoiding excessive sensitivity to minor temporal differences (Li & Xu, 2013).

**Stage 4: Composite Score Computation**

The final priority score is computed as the sum of hierarchical and freshness scores:

$$\text{Priority Score} = \text{Hierarchical Score} + \text{Freshness Score}$$

This additive model reflects the orthogonal nature of the scoring dimensions. The relative magnitudes are calibrated such that hierarchical scores substantially exceed maximum possible freshness scores, ensuring organizational applicability serves as the primary determinant. However, freshness scores remain sufficient to create meaningful differentiation among documents at equivalent hierarchical levels.

**Stage 5: Ranking and Selection**

Documents are sorted in descending order by composite score, producing a ranked list. The top-k highest-priority documents are selected for inclusion in the context provided to the language model. This selection ensures the language model receives the most organizationally relevant and temporally current information available, substantially improving response accuracy and applicability.

**Integration with Metadata Filtering**

The prioritization module operates in concert with the metadata filtering mechanism described in previous sections. While metadata filtering implements binary inclusion/exclusion based on applicability rules (course codes, degree programs, departments, faculties, batches, validity periods), the prioritization module provides graduated ranking among documents that pass filtering. This two-stage approach ensures that only applicable documents reach students, and among applicable documents, the most relevant appear first (Kaptein & Kamps, 2013).

**Handling Edge Cases and Exceptions**

The implementation includes provisions for several edge cases. When no documents at department or faculty levels pass filtering, university-level documents receive priority regardless of age, ensuring students receive some guidance. When multiple documents achieve identical composite scores, secondary tie-breaking occurs based on upload timestamps, with more recent documents preferred. For documents lacking upload date metadata, the system assigns a conservative default age resulting in reduced freshness scores, incentivizing proper metadata maintenance.

**Critical Application Scenarios**

The prioritization mechanism proves essential in several characteristic scenarios encountered in academic information systems:

*Policy Conflicts and Overrides:* Different organizational levels may establish potentially conflicting requirements. A university policy might establish a general deadline, while a faculty extends it, and a department further extends it. Hierarchical scoring ensures students receive the most specific applicable deadline rather than the general university provision, despite all documents being semantically similar.

*Departmental Specialization:* Different departments often establish distinct requirements for similar activities. Computer Science might require code repository submissions while Mathematics requires formal written proofs. Hierarchical scoring prevents cross-contamination of discipline-specific requirements.

*Administrative Procedure Variations:* Faculties with specialized characteristics often implement modified procedures. Medicine faculty with clinical rotations may have different registration processes than Arts faculty. Hierarchical scoring ensures students see procedures applicable to their context.

*Resource Access Questions:* University resources allocated at department or faculty levels with restricted access require appropriate prioritization. Queries about facilities, software, or services must return information about actually accessible resources rather than nominally available but inaccessible options.

*Temporal Sensitivity:* Academic calendars, policy revisions, contact information updates, course offering changes, and emergency communications all require freshness scoring to ensure students receive current information reflecting present institutional reality rather than outdated versions.

**Performance Considerations**

The prioritization module is designed for computational efficiency. Hierarchical classification operates in O(1) time per document, involving simple metadata field lookups. Score assignment similarly executes in constant time. Temporal analysis requires timestamp parsing and age calculation, also O(1) per document. The sorting operation dominates computational complexity at O(n log n) where n is the number of documents post-filtering, typically ranging from 20-50 documents. This computational profile ensures minimal latency impact on overall query response time.

**Advantages of the Dual-Factor Approach**

The integration of hierarchical and temporal scoring delivers several advantages over single-factor approaches. Purely hierarchical ranking without temporal consideration risks surfacing outdated department policies over current university-wide updates. Purely temporal ranking without hierarchical consideration risks prioritizing recent but inapplicable broad policies over older but specifically applicable department guidelines. The dual-factor approach balances these considerations, ensuring students receive information that is both organizationally appropriate and temporally current.

**Limitations and Mitigation Strategies**

The module exhibits dependency on metadata quality and completeness. Inconsistent or missing hierarchical level tags, organizational scope fields, or temporal metadata degrade prioritization quality. Mitigation strategies include implementing metadata validation at document upload, providing default values for missing fields, and establishing institutional documentation standards. Cross-cutting policies applying uniformly across organizational boundaries present challenges, as the module assumes department and faculty-level documents represent more specific guidance. This assumption holds for most policy categories but may inappropriately subordinate authoritative university-wide policies in certain domains such as codes of conduct or academic integrity policies that apply uniformly.

