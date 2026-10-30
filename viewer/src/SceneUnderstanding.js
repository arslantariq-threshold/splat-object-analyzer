// A scene-bound request panel. Late responses never replace a newly loaded scene.
export class SceneUnderstanding {
  constructor({onSelection, onView, onBusy}) {
    this.onSelection = onSelection;
    this.onView = onView;
    this.onBusy = onBusy;
    this.revision = 0;
    this.panel = document.getElementById('scene-understanding');
    this.form = document.getElementById('scene-prompt-form');
    this.input = document.getElementById('scene-prompt');
    this.submit = document.getElementById('scene-prompt-submit');
    this.state = document.getElementById('understanding-state');
    this.progress = document.getElementById('understanding-progress');
    this.summary = document.getElementById('scene-summary');
    this.groups = document.getElementById('scene-groups');
    this.selection = document.getElementById('subject-selection');
    this.views = document.getElementById('suggested-views');
    this.retry = document.getElementById('understanding-retry');
    this.retry.addEventListener('click', () => this.run(this.input.value.trim()));
    this.form.addEventListener('submit', event => {
      event.preventDefault();
      if (this.input.value.trim()) this.run(this.input.value.trim());
    });
    document.getElementById('show-all-subjects').addEventListener('click', () => {
      this.onSelection(null, this.index);
      this.selection.textContent = 'Showing all subjects allowed by the display settings.';
      this.views.replaceChildren();
    });
    document.getElementById('understanding-toggle').addEventListener('click', event => {
      const collapsed = this.panel.classList.toggle('collapsed');
      event.currentTarget.setAttribute('aria-expanded', String(!collapsed));
    });
    document.getElementById('download-understanding').addEventListener('click', () => {
      if (!this.index) return;
      const url = URL.createObjectURL(new Blob([JSON.stringify(this.index, null, 2)], {type:'application/json'}));
      const link = document.createElement('a');
      link.href = url; link.download = 'scene-understanding.json'; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
    this.reset();
  }

  reset() {
    if (this.busy) this.onBusy?.(false);
    this.revision++;
    this.pair = null;
    this.index = null;
    this.busy = false;
    this.panel.classList.add('hidden');
    for (const element of [this.summary, this.groups, this.selection, this.views]) element.replaceChildren();
    this.input.value = '';
  }

  setScene(pair) {
    this.reset();
    if (!pair?.sceneId || !pair.sourceHash) return;
    this.pair = pair;
    this.panel.classList.remove('hidden');
    this.run('');
  }

  status(message, busy = false, failed = false) {
    this.state.textContent = message;
    this.state.dataset.failed = String(failed);
    this.busy = busy;
    this.onBusy?.(busy);
    this.submit.disabled = busy;
    document.getElementById('download-understanding').disabled = !this.index;
    this.groups.querySelectorAll('button').forEach(button => { button.disabled = busy; });
    this.progress.classList.toggle('hidden', !busy);
    this.retry.classList.toggle('hidden', !failed);
  }

  async run(prompt) {
    if (!this.pair || this.busy) return;
    const revision = this.revision;
    this.status(prompt ? 'Finding subjects for your request…' : 'Understanding this scene…', true);
    this.views.replaceChildren();
    this.selection.replaceChildren();
    try {
      const response = await fetch('/api/understanding', {
        method:'POST', headers:{'Content-Type':'application/json', 'X-Splat-Viewer':'1'},
        body:JSON.stringify({sceneId:this.pair.sceneId, sourceHash:this.pair.sourceHash, prompt}),
      });
      const request = await response.json();
      if (!response.ok) throw new Error(request.error || 'Could not start scene understanding.');
      while (revision === this.revision) {
        const response = await fetch(`/api/understanding/${request.id}`);
        const job = await response.json();
        if (revision !== this.revision) return;
        if (!response.ok || job.status === 'failed') throw new Error(job.error || 'Scene understanding failed.');
        if (job.status === 'complete') {
          if (job.result.source_hash !== this.pair.sourceHash) throw new Error('The response belongs to another scene.');
          this.render(job.result);
          const review = this.index.review;
          const unresolved = review ? review.total - review.described_count : 0;
          this.status(unresolved ? `Ready · ${unresolved} descriptions unresolved; shown as uncertain` : 'Ready · descriptions are inferred from rendered views');
          return;
        }
        this.status(job.stage + (job.total ? ` · ${job.completed}/${job.total}` : ''), true);
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    } catch (error) {
      if (revision === this.revision) this.status(error.message, false, true);
    }
  }

  render(result) {
    this.index = result.understanding;
    if (!this.input.value && result.prompt) this.input.value = result.prompt;
    this.summary.replaceChildren();
    const type = document.createElement('strong');
    type.textContent = (this.index.scene.type || 'Scene').replaceAll('_', ' ');
    const description = document.createElement('p');
    description.textContent = this.index.scene.description;
    this.summary.append(type, description);
    this.groups.replaceChildren();
    for (const group of this.index.groups) {
      const button = document.createElement('button');
      button.type = 'button'; button.textContent = group.name;
      button.title = group.description;
      button.addEventListener('click', () => {
        this.input.value = `Show ${group.name}`;
        this.run(this.input.value);
      });
      this.groups.appendChild(button);
    }
    const heading = document.createElement('p');
    heading.textContent = result.selection.interpretation || 'Selected subjects';
    this.selection.appendChild(heading);
    const names = new Map(this.index.objects.map(object => [object.id, object]));
    const subjects = document.createElement('ul');
    for (const id of result.selection.object_ids) {
      const object = names.get(id);
      if (!object) continue;
      const item = document.createElement('li');
      item.textContent = object.display_name + (object.status === 'uncertain' ? ' (uncertain)' : '');
      subjects.appendChild(item);
    }
    this.selection.appendChild(subjects);
    if (result.selection.message || !result.selection.object_ids.length) {
      const message = document.createElement('p');
      message.textContent = result.selection.message || 'No supported matching subjects were found.';
      this.selection.appendChild(message);
    }
    this.onSelection(result.selection.object_ids, this.index);
    this.views.replaceChildren();
    for (const shot of result.shots) {
      const button = document.createElement('button');
      button.type = 'button'; button.className = 'suggested-view';
      const image = document.createElement('img');
      image.src = shot.imageUrl; image.alt = `Preview of ${shot.subjects.join(', ')}`;
      const label = document.createElement('span');
      label.textContent = shot.title;
      const detail = document.createElement('small');
      detail.textContent = shot.subjects.join(' · ');
      button.append(image, label, detail);
      button.addEventListener('click', () => {
        this.views.querySelectorAll('button').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
        this.onView(shot.camera, shot.object_ids);
        if (window.innerWidth < 780) {
          this.panel.classList.add('collapsed');
          document.getElementById('understanding-toggle').setAttribute('aria-expanded', 'false');
        }
      });
      button.setAttribute('aria-pressed', 'false');
      this.views.appendChild(button);
    }
    if (result.shots.length) {
      const note = document.createElement('p');
      note.className = 'view-note';
      note.textContent = 'Click a preview to inspect the view. These are existing analysis views; camera motion comes later.';
      this.views.appendChild(note);
    }
  }
}
