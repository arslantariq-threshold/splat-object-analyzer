// Annotation in viewer space. Coordinates are already mapped from the
// source JSON by AnnotationParser, so consumers use them as-is.
export class Annotation {
  constructor({ id = null, label, position, size, quaternion, color, parentId = null, kind = 'object', status = 'unreviewed' }) {
    this.id = id;
    this.label = label;
    this.originalLabel = label;
    this.resolvedLabel = label;
    this.kind = kind;
    this.status = status;
    this.originalStatus = status;
    this.position = position;   // THREE.Vector3
    this.size = size;           // THREE.Vector3 (full box extents)
    this.quaternion = quaternion; // THREE.Quaternion
    this.color = color;         // hex string
    this.parentId = parentId;
  }
}
