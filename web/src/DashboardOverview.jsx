function modalityName(value) {
  return ({ xray: "Chest X-ray", ct: "CT · Spleen", mri: "MRI · Brain" })[value] || value || "Imaging study";
}

function formatDate(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "Date unavailable" : date.toLocaleString();
}

function ServiceIndicator({ service }) {
  return (
    <div className="service-row">
      <span className={`service-indicator ${service.status}`} aria-hidden="true" />
      <div className="service-copy">
        <strong>{service.label}</strong>
        <span>{service.detail}</span>
      </div>
      <span className={`service-status ${service.status}`}>
        {service.status === "online" ? "Online" : service.status === "checking" ? "Checking" : "Offline"}
      </span>
    </div>
  );
}

export default function DashboardOverview({
  username,
  role,
  services,
  studies,
  studiesBusy,
  studiesError,
  onNavigate,
  onRefreshServices,
  onRefreshStudies,
}) {
  const serviceById = Object.fromEntries(services.map((service) => [service.id, service]));
  const modalities = [
    {
      id: "xray",
      title: "Chest X-ray",
      descriptor: "2D image research inference",
      service: serviceById.xray,
      note: "Raw finding scores and Grad-CAM sensitivity overlays when returned by the model.",
      access: "Research workflow",
    },
    {
      id: "ct",
      title: "CT · Spleen",
      descriptor: "3D NIfTI volume inspection",
      service: serviceById.ct,
      note: "Geometry inspection is separate from segmentation; signed orientation evidence is required.",
      access: "Pending independent review",
    },
    {
      id: "mri",
      title: "MRI · Brain",
      descriptor: "Four-channel BraTS workflow",
      service: serviceById.mri,
      note: "T1, T1c, T2 and FLAIR volumes for a task-specific research segmentation.",
      access: "Research workflow",
    },
  ];

  return (
    <div className="overview-page">
      <header className="page-heading overview-heading">
        <div>
          <p className="eyebrow">Research workspace</p>
          <h1>Good to see you, {username}</h1>
          <p className="header-subtitle">A focused workspace for imaging research and review.</p>
        </div>
        <div className="overview-heading-actions">
          <span className="research-chip"><span className="chip-dot" /> Research use only</span>
          <button type="button" className="button-secondary" onClick={onRefreshServices}>Refresh services</button>
        </div>
      </header>

      <section className="overview-banner">
        <div className="banner-mark" aria-hidden="true">+</div>
        <div>
          <p className="eyebrow">Medical Imaging AI</p>
          <h2>Imaging research, clearly organized.</h2>
          <p>Explore modality-specific workflows with their current availability and review safeguards in view.</p>
        </div>
        <div className="banner-account">
          <span className="avatar" aria-hidden="true">{username.slice(0, 1).toUpperCase()}</span>
          <span><strong>{username}</strong><small>{role}</small></span>
        </div>
      </section>

      <section className="overview-section" aria-labelledby="modalities-heading">
        <div className="section-title-row">
          <div>
            <p className="eyebrow">Workspaces</p>
            <h2 id="modalities-heading">Choose a modality</h2>
          </div>
          <span className="section-caption">Availability reflects API connectivity only</span>
        </div>
        <div className="modality-cards">
          {modalities.map((modality) => (
            <article className={`modality-card ${modality.id}`} key={modality.id}>
              <div className="modality-card-top">
                <span className="modality-symbol" aria-hidden="true">{modality.id === "xray" ? "XR" : modality.id.toUpperCase()}</span>
                <span className={`connection-tag ${modality.service?.status || "checking"}`}>
                  <span className={`service-indicator ${modality.service?.status || "checking"}`} />
                  {modality.service?.status === "online" ? "Connected" : modality.service?.status === "offline" ? "Offline" : "Checking"}
                </span>
              </div>
              <p className="modality-descriptor">{modality.descriptor}</p>
              <h3>{modality.title}</h3>
              <p className="modality-note">{modality.note}</p>
              <div className="modality-card-bottom">
                <span className={`access-label ${modality.id === "ct" ? "pending-review-label" : ""}`}
                  role={modality.id === "ct" ? "status" : undefined}>
                  {modality.access}
                </span>
                <button type="button" className="text-button" onClick={() => onNavigate(modality.id)}>
                  Open workspace <span aria-hidden="true">→</span>
                </button>
              </div>
            </article>
          ))}
        </div>
      </section>

      <div className="overview-lower-grid">
        <section className="surface-panel service-panel" aria-labelledby="services-heading">
          <div className="section-title-row">
            <div>
              <p className="eyebrow">System status</p>
              <h2 id="services-heading">API connections</h2>
            </div>
            <span className="section-caption">Live health checks</span>
          </div>
          <div className="service-list">
            {services.map((service) => <ServiceIndicator key={service.id} service={service} />)}
          </div>
          <p className="panel-footnote">Connection status does not indicate model validation, clinical suitability, or authorization.</p>
        </section>

        <section className="surface-panel recent-panel" aria-labelledby="recent-heading">
          <div className="section-title-row">
            <div>
              <p className="eyebrow">Review activity</p>
              <h2 id="recent-heading">Recent research sessions</h2>
            </div>
            <button type="button" className="text-button" onClick={() => onNavigate("sessions")}>
              View all <span aria-hidden="true">→</span>
            </button>
          </div>
          {studiesBusy ? (
            <p className="empty-state" role="status">Loading saved sessions…</p>
          ) : studiesError ? (
            <div className="inline-error" role="alert">
              <p>{studiesError}</p>
              <button type="button" className="button-secondary" onClick={onRefreshStudies}>Try again</button>
            </div>
          ) : studies.length ? (
            <ul className="recent-list">
              {studies.slice(0, 4).map((study) => (
                <li key={study.id || study.study_sha256}>
                  <span className="recent-modality">{(study.modality || "XR").slice(0, 2).toUpperCase()}</span>
                  <span className="recent-copy">
                    <strong>{modalityName(study.modality)}</strong>
                    <small>{formatDate(study.created_at)}</small>
                  </span>
                  <span className={`review-state ${study.review_status === "Reviewed" ? "complete" : ""}`}>
                    {study.review_status || study.ai_status || "Saved"}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <div className="empty-state">
              <span className="empty-mark" aria-hidden="true">◇</span>
              <strong>No saved sessions yet</strong>
              <p>Completed study records will appear here after they are saved by the review service.</p>
              <button type="button" className="text-button" onClick={() => onNavigate("xray")}>
                Start a research session <span aria-hidden="true">→</span>
              </button>
            </div>
          )}
        </section>
      </div>

      <p className="clinical-disclaimer">
        <strong>Research use only.</strong> Results are not clinically validated and must not be used for diagnosis or treatment.
      </p>
    </div>
  );
}
