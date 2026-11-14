# Publish mitsuba-video

The distribution is built from `packages/mitsuba_video`, which contains the general renderer. Source plugins
publish independently. `scripts/tf` in TextureFriction owns its recording plugin and `tf-render` command.

The `Publish mitsuba-video` workflow builds a wheel and source archive, checks their metadata, and tests the
wheel as a standalone install. Manual runs default to build-only. Publishing requires selecting `publish=true`
on the `main` branch and configuring the PyPI trusted publisher for:

- Project: `mitsuba-video`
- Repository owner: `ETSim`
- Repository: `MistubaVideo`
- Workflow: `publish.yml`
- Environment: `pypi`

Configure these values in the project's PyPI publishing settings, or as a pending publisher for a new project.
The [PyPI instructions](https://docs.pypi.org/trusted-publishers/adding-a-publisher/) describe how repository,
workflow and environment identify the publisher. No long-lived PyPI token is needed.

Update the version in `pyproject.toml` and `src/mitsuba_video/__init__.py` before a new release. Run the renderer
tests and notebook, then merge the reviewed changes. Trigger a build-only run to inspect the distributions
before publishing. Version 0.4 separates source plugins and defaults to geometry-only rendering; see
[the migration guide](plugins.md).

Adding this workflow does not register the project or publish a release. The repository owner must configure
the trusted publisher before the upload job can succeed.
