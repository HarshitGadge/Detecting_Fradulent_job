-- Rule-based screening baseline: each rule is a transparent, reviewable condition
-- that a trust & safety analyst could explain to a stakeholder.
-- Input table: postings (one row per job posting)
CREATE OR REPLACE TABLE rule_flags AS
SELECT
    job_id,
    fraudulent,
    (has_company_logo = 0 AND company_profile IS NULL)                        AS r1_no_logo_no_profile,
    (has_company_logo = 0 AND has_questions = 0)                              AS r2_no_logo_no_screening_qs,
    (industry IN ('Oil & Energy', 'Accounting'))                              AS r3_high_risk_industry,
    (telecommuting = 1 AND required_experience IN ('Entry level', 'Not Applicable')
        AND has_company_logo = 0)                                             AS r4_remote_entry_no_logo,
    (required_education IN ('High School or equivalent', 'Some High School Coursework')
        AND salary_range IS NOT NULL)                                         AS r5_low_edu_with_salary_bait,
    (regexp_matches(lower(coalesce(description, '') || ' ' || coalesce(benefits, '')),
        'work from home|earn \$|data entry|no experience (is )?(needed|required)|weekly pay'))
                                                                              AS r6_scam_phrasing
FROM postings;
