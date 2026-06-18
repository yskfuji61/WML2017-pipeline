# Data Policy — WMH2017 Local Research PoC

## Allowed data

- WMH2017 public Dataverse release under local `WMH2017_ROOT`
- Local generated manifests, split files, metrics JSON, and overlays

## Forbidden data

- private patient data
- customer data
- raw NIfTI files committed to git
- checkpoint or prediction mask committed to git
- unapproved cloud copies
- any dataset with unclear license or access terms

## Local path

Use:

```bash
export WMH2017_ROOT="<LOCAL_WMH2017_FILES_ROOT>"
```

The repository must never depend on this absolute path being committed into code. Store it in local shell, `.env.local`, or Cursor environment only.

## DLP boundary

This is a public challenge dataset, but it is still medical imaging data. Treat it as public-source, medical-domain, restricted-use research data.

## Stop conditions

Stop work and ask for review if:

- dataset root is not the Dataverse `files` layout
- terms/license are unclear
- private/PHI data appears
- cloud upload is needed
- test data is being used for train/val/tuning
- label values outside `{0,1,2}` appear
- raw NIfTI files appear in git status
